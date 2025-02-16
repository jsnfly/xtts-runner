import logging
import torch
import torchaudio
from pathlib import Path
from torch.functional import F

from gpt import XTTSGPT
from tokenizer import TextTokenizer

MODEL_DIR = Path(__file__).parent / "../XTTS-v2"
SPEAKER = "Aaron Dreschner"
LANG = "en"
TEXT = "Hi! This is a test!"

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    speaker_embeddings = torch.load(MODEL_DIR / "speakers_xtts.pth", weights_only=True)[SPEAKER]
    text_tokenizer = TextTokenizer(str(MODEL_DIR / "vocab.json"))

    gpt = XTTSGPT(MODEL_DIR / "config.json")
    gpt.load(MODEL_DIR / "model.pth")
    logging.debug("Model weights loaded")

    token_encoding = text_tokenizer.encode(TEXT, LANG)
    logging.debug(f"Text tokens: {token_encoding.tokens}")
    token_ids = torch.tensor(token_encoding.ids, dtype=torch.int32).unsqueeze(0)
    input_ids = gpt.compute_embeddings(speaker_embeddings["gpt_cond_latent"], token_ids)
    logging.debug(f"Input ids: {input_ids}")
    outputs = gpt.generate(
        input_ids,
        bos_token_id=gpt.config.gpt_start_audio_token,
        pad_token_id=gpt.config.gpt_stop_audio_token,
        eos_token_id=gpt.config.gpt_stop_audio_token,
        do_sample=False,
        top_p=0.85,
        top_k=50,
        temperature=0.75,
        num_return_sequences=1,
        num_beams=1,
        length_penalty=1.0,
        repetition_penalty=5.0,
        max_new_tokens=gpt.config.gpt_max_audio_tokens
    )
    logging.debug("Done generating")
    logging.debug(f"Outputs: {outputs}")
    text_emb = gpt.text_emb(token_ids) + gpt.text_pos_emb(token_ids)
    gpt_codes = outputs[:, input_ids.shape[1]:]
    code_stride_len = gpt.config.gpt_code_stride_len
    expected_output_len = gpt_codes.shape[-1] * code_stride_len
    code_lengths = torch.ceil(torch.tensor([expected_output_len]) / code_stride_len).long() + 3
    max_mel_len = code_lengths.max()
    audio_codes = F.pad(gpt_codes, (0, max_mel_len - gpt_codes.shape[-1]))
    audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=gpt.config.gpt_stop_audio_token)
    audio_codes[0, code_lengths[0] - 3:] = gpt.config.gpt_stop_audio_token
    audio_codes = F.pad(audio_codes, (1, 0), value=gpt.config.gpt_start_audio_token)
    audio_codes = F.pad(audio_codes, (0, 1), value=gpt.config.gpt_stop_audio_token)

    mel_emb = gpt.mel_emb(audio_codes) + gpt.mel_pos_emb(audio_codes)

    emb = torch.cat([speaker_embeddings["gpt_cond_latent"], text_emb, mel_emb], dim=1)

    gpt_out = gpt.super_forward(inputs_embeds=emb, return_dict=True)
    enc = gpt.final_norm(gpt_out.last_hidden_state[:, speaker_embeddings["gpt_cond_latent"].shape[1]:])
    mel_logits = enc[:, -mel_emb.shape[1]:]  # These are not really logits, but latents

    wav = gpt.decoder(mel_logits, g=speaker_embeddings["speaker_embedding"]).cpu().squeeze(0)
    torchaudio.save("test.wav", wav.detach(), sample_rate=24_000, format="wav")
    
