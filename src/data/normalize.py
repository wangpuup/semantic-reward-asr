"""Single text normalizer used for SFT targets, rewards and evaluation metrics."""
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("-", " ")
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()
