"""TinyWorld: a small synthetic language with an automatically checkable grammar.

Every sentence has the form ``<name> <verb> <count> <adj> <noun>.`` and the noun
must agree in number with the count (``one`` takes the singular form, every other
count takes the plural). Agreement is a dependency between distant character
positions, so a parallel decoder cannot satisfy it by getting characters right
independently.
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass

NAMES = ["ann", "bob", "cy", "dee", "eve", "fin", "gus", "ida"]
VERBS = ["sees", "has", "wants", "finds", "likes", "sells"]
COUNTS = ["one", "two", "three", "four", "five"]
ADJS = ["red", "big", "old", "shy", "odd", "wet", "tiny", "blue"]
NOUNS = {
    "cat": "cats",
    "dog": "dogs",
    "fox": "foxes",
    "box": "boxes",
    "hen": "hens",
    "bus": "buses",
    "cup": "cups",
    "egg": "eggs",
}

PAD = "_"
_SENTENCE = re.compile(r"^([a-z]+) ([a-z]+) ([a-z]+) ([a-z]+) ([a-z]+)\.$")
_PLURAL_TO_SINGULAR = {plural: singular for singular, plural in NOUNS.items()}


def all_sentences() -> list[str]:
    out = []
    for name, verb, count, adj, noun in itertools.product(NAMES, VERBS, COUNTS, ADJS, NOUNS):
        form = noun if count == "one" else NOUNS[noun]
        out.append(f"{name} {verb} {count} {adj} {form}.")
    return out


def max_sentence_length() -> int:
    return max(len(s) for s in all_sentences())


@dataclass(frozen=True)
class Check:
    well_formed: bool
    agrees: bool

    @property
    def valid(self) -> bool:
        return self.well_formed and self.agrees


def check(text: str) -> Check:
    """Parse one generated string. Trailing padding is ignored."""
    body = text.rstrip(PAD)
    m = _SENTENCE.match(body)
    if not m:
        return Check(False, False)
    name, verb, count, adj, noun = m.groups()
    if name not in NAMES or verb not in VERBS or count not in COUNTS or adj not in ADJS:
        return Check(False, False)
    if noun in NOUNS:
        plural = False
    elif noun in _PLURAL_TO_SINGULAR:
        plural = True
    else:
        return Check(False, False)
    return Check(True, plural == (count != "one"))


def split(seed: int, n_train: int, n_val: int) -> tuple[list[str], list[str], list[str]]:
    """Deterministic disjoint train / validation / unseen-test split of the whole language."""
    sentences = all_sentences()
    rng = random.Random(seed)
    rng.shuffle(sentences)
    if n_train + n_val > len(sentences):
        raise ValueError(f"language has only {len(sentences)} sentences")
    return sentences[:n_train], sentences[n_train : n_train + n_val], sentences[n_train + n_val :]
