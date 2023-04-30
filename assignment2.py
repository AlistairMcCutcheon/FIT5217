from __future__ import unicode_literals, print_function, division, annotations
from io import open
import unicodedata
import string
import re
import random
from unidecode import unidecode
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import string
from torch.utils.data import Dataset
from glob import glob
from pathlib import Path

from typing import Iterable, Generator


SOS_TOKEN = 0
EOS_TOKEN = 1


class CookingDataset(Dataset):
    def __init__(self, data_dir_path: str) -> None:
        super().__init__()

        self.ingredients, self.recepies = read_ingredients_recepies(data_dir_path)
        self.lang = self.create_lang()
        print(
            self.lang.word2index,
            self.lang.word2count,
            self.lang.n_words,
        )
        print(get_alphabet(self.ingredients).union(get_alphabet(self.recepies)))

    def create_lang(self) -> Lang:
        lang = Lang()
        for text in self.ingredients:
            lang.add_sentence(text)
        for text in self.recepies:
            lang.add_sentence(text)
        return lang

    def summarise(self):
        print("Ingredients Alphabet:")
        print(get_alphabet(self.ingredients))
        print("Recepies Alphabet")
        print(get_alphabet(self.recepies))
        pass


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word: str) -> None:
        if word in self.word2index:
            self.word2count[word] += 1
            return

        self.word2index[word] = self.n_words
        self.word2count[word] = 1
        self.index2word[self.n_words] = word
        self.n_words += 1


def remove_successive_chars(s: str, chars_to_keep_unchanged: set[str]):
    previous_char = None
    new_string = []
    for char in s:
        if char in chars_to_keep_unchanged:
            new_string.append(char)
            previous_char = char
            continue
        if char == previous_char:
            continue
        new_string.append(char)
        previous_char = char
    return "".join(new_string)


def replace(s: str, chars_to_replace: set[str], string_to_replace_with: str) -> str:
    new_string = []
    for char in s:
        if char in chars_to_replace:
            new_string.append(string_to_replace_with)
            continue
        new_string.append(char)
    return "".join(new_string)


def keep_chars(s, chars_to_keep: set[str]):
    return "".join([c for c in s if c in chars_to_keep])


def normalise_string(s: str):
    s = s.lower().strip()
    s = unidecode(s)
    s = remove_successive_chars(
        s, {*list(string.ascii_lowercase), *list(string.digits)}
    )
    s = replace(s, {"*", "|", "\x00", ":", "_", "\x7f", "\n", "\\", "\t"}, " ")
    s = s.replace('"', "''")
    s = s.replace("@", " at ")
    s = s.replace("-", " - ")
    s = s.replace(",", " , ")
    s = s.replace(".", " . ")
    s = replace(s, {"&", "+"}, " and ")
    s = replace(s, {"!", "?"}, ".")
    s = keep_chars(
        s, {*list(string.ascii_lowercase), *list(string.digits), "-", ".", ",", " "}
    )
    s = remove_successive_chars(
        s, {*list(string.ascii_lowercase), *list(string.digits)}
    )
    return s


def read_ingredient_recipie(
    text_lines: Iterable[str],
) -> Generator[tuple[str, str], None, None]:
    for line in text_lines:
        if not line.startswith("ingredients:"):
            continue

        ingredients = line.removeprefix("ingredients:")
        recepie = read_next_recepie(text_lines)
        yield ingredients, recepie


def read_next_recepie(text_lines: Iterable[str]) -> str:
    recepie_lines = []
    for line in text_lines:
        if line.startswith("END RECIPE"):
            break
        recepie_lines.append(line)
    return "".join(recepie_lines)


def read_ingredients_recepies(data_dir_path: str):
    ingredients_per_recepie = []
    recepies = []
    i = 0
    for recipies_file_path in Path(data_dir_path).glob("*"):
        with open(recipies_file_path, "r") as file:
            for ingredients, recepie in read_ingredient_recipie(file):
                ingredients_per_recepie.append(normalise_string(ingredients))
                recepies.append(normalise_string(recepie))
        if i >= 5:
            break
        i += 1
    return ingredients_per_recepie, recepies


def get_alphabet(text_lines: Iterable[str]) -> set[str]:
    alphabet = set()
    for text_line in text_lines:
        alphabet.update(set(text_line))
    return alphabet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x = CookingDataset("data/train")
