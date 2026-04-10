# clean text
import re
from pyvi import ViTokenizer
from typing import List, Set

CLEAN_PATTERN = re.compile(
    r"http\S+|@\S+|#\S+|\d+"
)

SPECIAL_PATTERN = re.compile(r"[^a-zA-ZÀ-ỹ!?\\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
def clean_text(text: str) -> str:
    text = text.lower()
    text = CLEAN_PATTERN.sub(" ", text)
    text = SPECIAL_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text

def tokenize(text: str) -> list:
    return ViTokenizer.tokenize(text).split()

def load_stopwords(path: str) -> set:
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f)

def remove_stopwords(tokens: List[str], stopwords: Set[str]) -> List[str]:
    return [word for word in tokens if word not in stopwords]

def preprocess(text: str, stopwords: Set[str] = None) -> List[str]:
    # clean text
    text = clean_text(text)
    # tokenize text
    tokens = tokenize(text)
    # remove stopwords
    if stopwords is not None:
        tokens = remove_stopwords(tokens, stopwords)
    
    return tokens
    