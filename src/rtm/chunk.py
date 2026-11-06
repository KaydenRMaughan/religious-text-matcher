"""Split long texts into passages that fit an embedding model's token limit.

Talks are one block of ~1,700 words with no paragraph breaks, so passages are built
from sentences: pack whole sentences into windows of at most ``max_words`` words,
repeating the last ``overlap`` sentences at the start of the next window so a
thought that straddles a boundary is seen whole at least once.

At ~1.23 mpnet tokens per word, 200 words is ~250 tokens: inside MiniLM's 256 and
mpnet's 384. A document is then scored by its best passage (max-pooling).
"""
from __future__ import annotations

import re

# A sentence ends at . ! or ? (optionally followed by a closing quote or bracket),
# then whitespace, then an uppercase letter, digit or opening quote.
_SENTENCE_END = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))\s+(?=[\"'“‘(\[A-Z0-9])"
)


def split_sentences(text: str) -> list[str]:
    """Rough sentence split. Good enough for windowing; not a linguistic parser."""
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def chunk_text(text: str, max_words: int = 200, overlap: int = 1) -> list[str]:
    """Pack sentences into windows of at most ``max_words`` words, with ``overlap``
    sentences repeated between consecutive windows.

    A single sentence longer than ``max_words`` is split on word boundaries. A text
    shorter than ``max_words`` comes back as one chunk. Empty text gives [].
    """
    if max_words < 1 or overlap < 0:
        raise ValueError("max_words must be >= 1 and overlap >= 0")
    sentences: list[list[str]] = []
    for s in split_sentences(text):
        words = s.split()
        sentences += [words[i:i + max_words] for i in range(0, len(words), max_words)]
    chunks, start = [], 0
    while start < len(sentences):
        end, n = start, 0
        while end < len(sentences) and (n + len(sentences[end]) <= max_words or end == start):
            n += len(sentences[end])
            end += 1
        chunks.append(" ".join(w for s in sentences[start:end] for w in s))
        if end >= len(sentences):
            break
        start = max(end - overlap, start + 1)  # always advance
    return chunks
