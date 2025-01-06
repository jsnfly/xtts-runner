import re
from tokenizers import Tokenizer


class TextTokenizer:
    CHAR_LIMITS = {
        "en": 250,
        "de": 253,
        "fr": 273,
        "es": 239,
        "it": 213,
        "pt": 203,
        "pl": 224,
        "zh": 82,
        "ar": 166,
        "cs": 186,
        "ru": 182,
        "nl": 251,
        "tr": 226,
        "ja": 71,
        "hu": 224,
        "ko": 95,
        "hi": 150,
    }

    def __init__(self, vocab_file):
        # `Tokenizer` can not be subclassed so it has to be handled like this.
        self.tokenizer = Tokenizer.from_file(str(vocab_file))

    def encode(self, text, lang):
        lang = lang.split("-")[0]  # remove the region
        self.check_input_length(text, lang)
        text = self.preprocess_text(text)
        text = f"[{lang}]{text}"
        text = text.replace(" ", "[SPACE]")
        return self.tokenizer.encode(text)

    def check_input_length(self, text, lang):
        limit = self.CHAR_LIMITS.get(lang, 250)
        if len(text) > limit:
            print(
                f"The text length exceeds the character limit of {limit} for language '{lang}', "
                "this might cause truncated audio."
            )

    def preprocess_text(self, text):
        # Original has some more stuff, but seems incomplete even there.
        return re.sub(r"\s+", " ", text.lower().replace('"', ""))
