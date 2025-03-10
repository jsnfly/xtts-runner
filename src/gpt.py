import json
import random
import re
import torch
import zipfile

from functools import partial
from torch import nn
from torch.functional import F
from transformers import GPT2Config, GPT2Model
from transformers import GenerationMixin, GPT2Model

from custom_unpickler import CustomUnpickler
from hifi_decoder import HifiDecoder


def null_position_embeddings(range_, dim):
    return torch.zeros((range_.shape[0], range_.shape[1], dim), device=range_.device)


# TODO: is this class necessary?
class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        # nn.Embedding
        self.emb = torch.nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class XTTSGPT(GPT2Model, GenerationMixin):

    def __init__(self, config_path):
        config = self.load_config(config_path)

        max_seq_len = sum(
            config[key] for key in ["gpt_max_audio_tokens", "gpt_max_text_tokens", "gpt_max_prompt_tokens"]
        ) + 1

        # Sets `self.config`` to this config.
        super().__init__(GPT2Config(
            vocab_size=config["gpt_max_audio_tokens"],
            n_positions=max_seq_len,
            n_ctx=max_seq_len,
            n_embd=config["gpt_n_model_channels"],
            n_layer=config["gpt_layers"],
            n_head=config["gpt_n_heads"],
            **config
        ))

        self.text_emb = nn.Embedding(self.config.gpt_number_text_tokens, self.config.n_embd)
        self.text_pos_emb = LearnedPositionEmbeddings(self.config.gpt_max_text_tokens + 2, self.config.n_embd)
        self.mel_emb = nn.Embedding(self.config.gpt_num_audio_tokens, self.config.n_embd)
        self.mel_pos_emb = LearnedPositionEmbeddings(self.config.gpt_max_audio_tokens + 2 + 1, self.config.n_embd)

        # Built-in token embeddings are unused.
        del self.wpe
        self.wpe = partial(null_position_embeddings, dim=self.config.n_embd)
        del self.wte

        self.cond_latent = self.speaker_emb = None

        self.final_norm = nn.LayerNorm(self.config.n_embd)
        self.mel_head = nn.Linear(self.config.n_embd, self.config.gpt_num_audio_tokens)
        self.decoder = HifiDecoder()

    @staticmethod
    def load_config(config_path):
        with open(config_path) as f:
            return json.load(f)["model_args"]


    def load(self, model_path):
        with zipfile.ZipFile(model_path, 'r') as archive:
            with archive.open('model/data.pkl', 'r') as f:
                data = CustomUnpickler(f, archive).load()
        state_dict = {re.sub(r"^gpt\.gpt\.", "", key): value for key, value in data['model'].items()
                      if key.startswith("gpt.gpt.")}
        state_dict |= {re.sub(r"^hifigan_", "", key): value for key, value in data['model'].items()
                       if key.startswith("hifigan_decoder.")}
        state_dict['text_emb.weight'] = data['model']['gpt.text_embedding.weight']
        state_dict['text_pos_emb.emb.weight'] = data['model']['gpt.text_pos_embedding.emb.weight']
        state_dict['mel_emb.weight'] = data['model']['gpt.mel_embedding.weight']
        state_dict['mel_pos_emb.emb.weight'] = data['model']['gpt.mel_pos_embedding.emb.weight']
        state_dict['final_norm.weight'] = data['model']['gpt.final_norm.weight']
        state_dict['final_norm.bias'] = data['model']['gpt.final_norm.bias']
        state_dict['mel_head.weight'] = data['model']['gpt.mel_head.weight']
        state_dict['mel_head.bias'] = data['model']['gpt.mel_head.bias']
        self.load_state_dict(state_dict)

        return self.eval()

    def set_speaker_embeddings(self, speakers_path, speaker):
        emb_dict = torch.load(speakers_path, weights_only=True)[speaker]
        self.cond_latent = emb_dict["gpt_cond_latent"].to(self.device)
        self.speaker_emb = emb_dict["speaker_embedding"].to(self.device)

    @torch.no_grad()
    def generate(self, input_ids, all_latents, *args, **kwargs):
        if self.cond_latent is None:
            raise ValueError("Latent conditionals are not set, please use `#set_speaker_embeddings` first.")

        # We can't pass the text token IDs to `GenerationMixin#generate`, as some `LogitsProcessor` will use them and
        # that might lead to errors, as the model is only supposed to generate audio token ids. However, we need the
        # text embeddings for the first step of generation, so we calculate it here.
        text_token_ids = input_ids[:, :-1]
        text_emb = self.text_emb(text_token_ids) + self.text_pos_emb(text_token_ids)
        self.prefix_emb = torch.cat([self.cond_latent, text_emb], dim=1)

        return super().generate(input_ids[:, -1:], all_latents=all_latents, *args, **kwargs)


    def forward(self, input_ids, attention_mask, position_ids, all_latents, past_key_values=None, *args, **kwargs):
        emb = self.mel_emb(input_ids[:, -1]) + self.mel_pos_emb.get_fixed_embedding(position_ids[:, -1],
                                                                                    input_ids.device)
        if input_ids[0][-1] == self.config.gpt_start_audio_token:
            # Only for first generation step.
            emb = torch.cat([self.prefix_emb, emb], dim=1)

        outputs = super().forward(inputs_embeds=emb, attention_mask=attention_mask, position_ids=position_ids,
                                  past_key_values=past_key_values, use_cache=True, output_hidden_states=True)

        latents = self.final_norm(outputs.hidden_states[-1])
        all_latents.append(latents)
        outputs.hidden_states = tuple()  # Not required as the last hidden state was already stored in `all_latents`.
        outputs.logits = self.mel_head(latents)
        return outputs
