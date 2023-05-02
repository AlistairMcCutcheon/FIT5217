from __future__ import unicode_literals, print_function, division, annotations
from io import open
import string
from unidecode import unidecode
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import (
    pack_sequence,
    PackedSequence,
    pad_packed_sequence,
    pack_padded_sequence,
)

import torch.nn.functional as F
import string
from torch.utils.data import Dataset
from pathlib import Path
from torch import Tensor
import statistics
from typing import Any, Iterable, Generator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN = 0
EOS_TOKEN = 1


class CookingDataset(Dataset):
    def __init__(self, data_dir_path: str, max_len: int = 150) -> None:
        super().__init__()

        self.ingredients, self.recipes = read_ingredients_recepies(data_dir_path)

        self.lang = create_lang(self.ingredients, self.recipes)

        tokenised_ingredients = [
            indexes_from_text(self.lang, x, sos_token=False, eos_token=True)
            for x in self.ingredients
        ]
        tokenised_recipes = [
            indexes_from_text(self.lang, x, sos_token=True, eos_token=True)
            for x in self.recipes
        ]

        indexes = {
            i
            for i, (x, y) in enumerate(zip(tokenised_ingredients, tokenised_recipes))
            if len(x) <= 150 and len(y) <= max_len
        }

        self.ingredients = [self.ingredients[x] for x in indexes]
        self.recipes = [self.recipes[x] for x in indexes]
        self.tokenised_ingredients = [tokenised_ingredients[x] for x in indexes]
        self.tokenised_recipes = [tokenised_recipes[x] for x in indexes]

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        ingredients_tensor = torch.tensor(
            self.tokenised_ingredients[index], dtype=torch.long, device=DEVICE
        )
        recepe_tensor = torch.tensor(
            self.tokenised_recipes[index], dtype=torch.long, device=DEVICE
        )
        return ingredients_tensor, recepe_tensor

    def __len__(self):
        return len(self.ingredients)

    def summarise(self):
        ingredients_alphabet = get_alphabet(self.ingredients)
        print("Ingredients Alphabet:")
        print(ingredients_alphabet)
        print("Ingredients Alphabet Length:")
        print(len(ingredients_alphabet))
        print()

        recepies_alphabet = get_alphabet(self.recipes)
        print("Recepies Alphabet:")
        print(recepies_alphabet)
        print("Recepies Alphabet Length:")
        print(len(recepies_alphabet))
        print()

        print("Sample Ingredients:")
        print(self.ingredients[0])
        print("Sample Recepie:")
        print(self.recipes[0])
        print()

        print("Number of Samples:")
        print(len(self))
        print()

        print("Vocabulary Size, including SOS_TOKEN and EOS_TOKEN:")
        print(self.lang.n_words)
        print()

        (
            ingredients_min_len,
            ingredients_max_len,
            ingredients_avg_len,
            ingredients_stdev_len,
            ingredients_median_len,
        ) = summarise_tokens([x[0] for x in self])

        print(f"Length of shortest ingredients text: {ingredients_min_len}")
        print(f"Length of longest ingredients text: {ingredients_max_len}")
        print(f"Average length of ingredients text: {ingredients_avg_len}")
        print(
            f"Standard deviation of lengths of ingredients text: {ingredients_stdev_len}"
        )
        print(f"Median of lengths of ingredients text: {ingredients_median_len}")

        (
            recepie_min_len,
            recepie_max_len,
            recepie_avg_len,
            recepie_stdev_len,
            recepie_median_len,
        ) = summarise_tokens([x[1] for x in self])
        print()

        print(f"Length of shortest recepie text: {recepie_min_len}")
        print(f"Length of longest recepie text: {recepie_max_len}")
        print(f"Average length of recepie text: {recepie_avg_len}")
        print(f"Standard deviation of lengths of recepie text: {recepie_stdev_len}")
        print(f"Median of lengths of recepie text: {recepie_median_len}")
        print()


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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input: PackedSequence):
        seq_unpacked, lens_unpacked = pad_packed_sequence(input, batch_first=True)
        batch_size = len(seq_unpacked)
        embedded = self.embedding(seq_unpacked)
        embedded = pack_padded_sequence(
            embedded, lens_unpacked, batch_first=True, enforce_sorted=False
        )

        hidden_0 = torch.zeros((1, batch_size, self.hidden_size), device=DEVICE)
        output, hidden = self.gru(embedded, hidden_0)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, target: PackedSequence, hidden):
        seq_unpacked, lens_unpacked = pad_packed_sequence(target, batch_first=True)
        embedded = self.embedding(seq_unpacked)

        output = self.dropout(embedded)
        output = F.relu(embedded)
        output = pack_padded_sequence(
            output, lens_unpacked, batch_first=True, enforce_sorted=False
        )

        output, hidden = self.gru(output, hidden)

        seq_unpacked, lens_unpacked = pad_packed_sequence(output, batch_first=True)

        output = self.out(seq_unpacked)
        output = self.softmax(output)
        output = pack_padded_sequence(
            output, lens_unpacked, batch_first=True, enforce_sorted=False
        )
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_len=150):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, max_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        print(input.shape)
        print(hidden.shape)
        print(encoder_outputs.shape)
        batch_size, sequence_length = input.shape
        embedded = self.embedding(input).view(
            sequence_length, batch_size, self.hidden_size
        )
        embedded = self.dropout(embedded)

        print(embedded.shape)

        features = torch.cat((embedded[0], hidden[0]), 1)
        print(features.shape)
        features = self.attn(features)
        attention_weights = F.softmax(features, dim=1)

        print(attention_weights.shape)
        print(encoder_outputs.shape)

        attn_applied = torch.bmm(
            attention_weights.unsqueeze(0), encoder_outputs[0].unsqueeze(0)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


class SequenceNLLLoss(nn.Module):
    def forward(self, input: PackedSequence, target: PackedSequence) -> Any:
        input_seq_unpacked, input_seq_lengths = pad_packed_sequence(
            input, batch_first=True
        )
        target_seq_unpacked, _ = pad_packed_sequence(target, batch_first=True)

        loss = 0
        for sequence, target_sequence, input_seq_length in zip(
            input_seq_unpacked, target_seq_unpacked, input_seq_lengths
        ):
            loss += (
                F.nll_loss(
                    sequence[:input_seq_length],
                    target_sequence[:input_seq_length],
                    reduction="sum",
                )
                / input_seq_length
            )
        return loss


class SeqToSeq(nn.Module):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target):
        _, encoder_hidden = encoder(input)
        decoder_output, _ = decoder(target, encoder_hidden)
        return decoder_output


class SeqToSeqWithAttention(nn.Module):
    def __init__(self, encoder: EncoderRNN, decoder: AttnDecoderRNN) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target):
        encoder_outputs, encoder_hidden = encoder(input)
        decoder_output, _, _ = decoder(target, encoder_hidden, encoder_outputs)
        return decoder_output


def create_lang(ingredients: list[str], recipes: list[str]) -> Lang:
    lang = Lang()
    for text in ingredients:
        lang.add_sentence(text)
    for text in recipes:
        lang.add_sentence(text)
    return lang


def summarise_tokens(tokens: list[list[int]]) -> tuple[int, int, float]:
    lengths = [len(x) for x in tokens]
    min_length = min(lengths)
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)
    stdev_length = statistics.stdev(lengths)
    median_length = statistics.median(lengths)

    return min_length, max_length, avg_length, stdev_length, median_length


def indexes_from_text(
    lang: Lang, text: str, sos_token: bool = False, eos_token: bool = False
):
    indexes = [lang.word2index[word] for word in text.split(" ")]
    if sos_token:
        indexes.insert(0, SOS_TOKEN)
    if eos_token:
        indexes.append(EOS_TOKEN)
    return indexes


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


def read_ingredient_recipe(
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
    for recipes_file_path in Path(data_dir_path).glob("*"):
        with open(recipes_file_path, "r") as file:
            for ingredients, recepie in read_ingredient_recipe(file):
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


def train(
    dataloader: DataLoader,
    seq_to_seq,
    optimiser: Optimizer,
    criterion,
    writer: SummaryWriter,
):
    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        # batch_size = len(input_tensor)
        # batch_size = input_tensor.shape[0]
        seq_to_seq.zero_grad()

        # input_length = input_tensor.shape[-1]
        # target_length = target_tensor.shape[-1]

        output = seq_to_seq.forward(input_tensor, target_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()

        optimiser.step()
        writer.add_scalar("Loss", loss.item(), i)


def collate_fn(input_batch):
    # max_ingredients_length = max((len(ingredients) for ingredients, _ in input_batch))
    # max_recipe_length = max((len(recepe) for _, recepe in input_batch))

    # ingredients_batch = torch.full(
    #     (len(input_batch), max_ingredients_length),
    #     fill_value=EOS_TOKEN,
    #     dtype=torch.long,
    #     device=DEVICE,
    # )
    # recipes_batch = torch.full(
    #     (len(input_batch), max_recipe_length),
    #     fill_value=EOS_TOKEN,
    #     dtype=torch.long,
    #     device=DEVICE,
    # )

    # for i, (ingredients, recipe) in enumerate(input_batch):
    #     ingredients_batch[i, : len(ingredients)] = ingredients
    #     recipes_batch[i, : len(recipe)] = recipe
    ingredients_batch = pack_sequence(
        [ingredients for ingredients, _ in input_batch], enforce_sorted=False
    )
    recipies_batch = pack_sequence(
        [recipe for _, recipe in input_batch], enforce_sorted=False
    )

    return ingredients_batch, recipies_batch


dataset = CookingDataset("data/train")
dataset.summarise()
dataloader = DataLoader(dataset, 32, shuffle=True, collate_fn=collate_fn)

hidden_size = 256
encoder = EncoderRNN(dataset.lang.n_words, hidden_size).to(DEVICE)

decoder = DecoderRNN(hidden_size, dataset.lang.n_words).to(DEVICE)
seq_to_seq = SeqToSeq(encoder, decoder)

# decoder = AttnDecoderRNN(hidden_size, dataset.lang.n_words).to(DEVICE)
# seq_to_seq = SeqToSeqWithAttention(encoder, decoder)

writer = SummaryWriter()

learning_rate = 0.001
optimiser = optim.Adam(seq_to_seq.parameters(), lr=learning_rate)
# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = SequenceNLLLoss()

train(
    dataloader,
    seq_to_seq,
    optimiser,
    criterion,
    writer,
)
