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
    text_tokenizer = TextTokenizer(str(MODEL_DIR / "vocab.json"))

    gpt = XTTSGPT(MODEL_DIR / "config.json")
    gpt.load(MODEL_DIR / "model.pth")
    gpt.set_speaker_embeddings(MODEL_DIR / "speakers_xtts.pth", SPEAKER)
    logging.debug("Model weights loaded")

    token_encoding = text_tokenizer.encode(TEXT, LANG)
    logging.debug(f"Text tokens: {token_encoding.tokens}")
    input_ids = torch.tensor(token_encoding.ids + [gpt.config.gpt_start_audio_token], dtype=torch.int64).unsqueeze(0)

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
        max_new_tokens=gpt.config.gpt_max_audio_tokens,
        return_dict_in_generate=True,
        output_hidden_states=True
    )
    logging.debug("Done generating")
    last_hidden_states = torch.cat([h[-1] for h in outputs.hidden_states], dim=1)[:, -outputs.sequences.shape[1]:]
    wav = gpt.decoder(last_hidden_states, g=gpt.speaker_emb).cpu().squeeze(0)
    torchaudio.save("test.wav", wav.detach(), sample_rate=24_000, format="wav")
