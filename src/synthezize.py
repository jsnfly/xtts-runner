import json
import logging
import torch
from pathlib import Path

from tokenizer import TextTokenizer

MODEL_PATH = Path("../XTTS-v2")
SPEAKER = "Aaron Dreschner"
LANG = "en"
TEXT = "Hi! This is a test!"

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    with open(MODEL_PATH / "config.json") as f:
        config = json.load(f)

    speaker_embeddings = torch.load(MODEL_PATH / "speakers_xtts.pth", weights_only=True)[SPEAKER]
    text_tokenizer = TextTokenizer(str(MODEL_PATH / "vocab.json"))
    logging.debug(f"Text tokens: {text_tokenizer.encode(TEXT, LANG).tokens}")
