import logging
import torch
import torchaudio
from pathlib import Path
from torch.functional import F

from gpt import XTTSGPT
from tokenizer import TextTokenizer
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main(model_dir: str, text: str, lang: str, speaker: str, output_file: str):
    text_tokenizer = TextTokenizer(str(Path(model_dir) / "vocab.json"))

    gpt = XTTSGPT(Path(model_dir) / "config.json")
    gpt.load(Path(model_dir) / "model.pth")
    gpt.to(DEVICE)
    gpt.set_speaker_embeddings(Path(model_dir) / "speakers_xtts.pth", speaker)
    logging.debug("Model weights loaded")

    token_encoding = text_tokenizer.encode(text, lang)
    logging.debug(f"Text tokens: {token_encoding.tokens}")
    input_ids = torch.tensor(token_encoding.ids + [gpt.config.gpt_start_audio_token], dtype=torch.int64).unsqueeze(0)

    logging.debug(f"Input ids: {input_ids}")
    outputs = gpt.generate(
        input_ids.to(DEVICE),
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
        max_new_tokens=gpt.config.gpt_max_audio_tokens,
        return_dict_in_generate=True,
        output_hidden_states=True
    )
    logging.debug("Done generating")
    last_hidden_states = torch.cat([h[-1] for h in outputs.hidden_states], dim=1)[:, -outputs.sequences.shape[1]:]
    wav = gpt.decoder(last_hidden_states, g=gpt.speaker_emb).cpu().squeeze(0)
    torchaudio.save(output_file, wav.detach(), sample_rate=24_000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthesize speech from text using XTTSGPT model.')
    parser.add_argument('model_dir', type=str, help='Path to the directory containing the model files.')
    parser.add_argument('--text', type=str, required=True, help='The text to synthesize speech from.')
    parser.add_argument('--lang', type=str, default="en", help='The language of the input text. Default is "en".')
    parser.add_argument('--speaker', type=str, default="Aaron Dreschner", help='Speaker voice.')
    parser.add_argument('--output_file', type=str, help='The file to store the generated audio in.')

    args = parser.parse_args()
    main(args.model_dir, args.text, args.lang, args.speaker, args.output_file)
