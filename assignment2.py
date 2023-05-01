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
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import matplotlib.ticker as ticker
import numpy as np
import torch.nn.functional as F
import string
from torch.utils.data import Dataset
from glob import glob
from pathlib import Path
from torch import Tensor
import statistics
from typing import Iterable, Generator
from torch.optim import Optimizer
import time
import math
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN = 0
EOS_TOKEN = 1
TEACHER_FORCING_RATIO = 1


class CookingDataset(Dataset):
    def __init__(self, data_dir_path: str) -> None:
        super().__init__()

        self.ingredients, self.recepies = read_ingredients_recepies(data_dir_path)
        self.lang = self.create_lang()
        self.summarise()

    def create_lang(self) -> Lang:
        lang = Lang()
        for text in self.ingredients:
            lang.add_sentence(text)
        for text in self.recepies:
            lang.add_sentence(text)
        return lang

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        ingredients_tensor = tensor_from_text(self.lang, self.ingredients[index])
        recepie_tensor = tensor_from_text(self.lang, self.recepies[index])
        return ingredients_tensor, recepie_tensor

    def __len__(self):
        return len(self.ingredients)

    def summarise(self):
        ingredients_alphabet = get_alphabet(self.ingredients)
        print("Ingredients Alphabet:")
        print(ingredients_alphabet)
        print("Ingredients Alphabet Length:")
        print(len(ingredients_alphabet))
        print()

        recepies_alphabet = get_alphabet(self.recepies)
        print("Recepies Alphabet:")
        print(recepies_alphabet)
        print("Recepies Alphabet Length:")
        print(len(recepies_alphabet))
        print()

        print("Sample Ingredients:")
        print(self.ingredients[0])
        print("Sample Recepie:")
        print(self.recepies[0])
        print()

        print("Number of Samples:")
        print(len(self))
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
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, input.shape[0], -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, input.shape[0], -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


def summarise_tokens(tokens: list[list[int]]) -> tuple[int, int, float]:
    lengths = [len(x) for x in tokens]
    min_length = min(lengths)
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)
    stdev_length = statistics.stdev(lengths)
    median_length = statistics.median(lengths)

    return min_length, max_length, avg_length, stdev_length, median_length


def indexes_from_text(lang: Lang, text: str):
    return [lang.word2index[word] for word in text.split(" ")]


def tensor_from_text(lang: Lang, text: str):
    indexes = indexes_from_text(lang, text)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE)  # .view(-1, 1)


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
    # i = 0
    for recipes_file_path in Path(data_dir_path).glob("*"):
        with open(recipes_file_path, "r") as file:
            for ingredients, recepie in read_ingredient_recipe(file):
                ingredients_per_recepie.append(normalise_string(ingredients))
                recepies.append(normalise_string(recepie))
        # if i >= 5:
        #     break
        # i += 1
    return ingredients_per_recepie, recepies


def get_alphabet(text_lines: Iterable[str]) -> set[str]:
    alphabet = set()
    for text_line in text_lines:
        alphabet.update(set(text_line))
    return alphabet


def train(
    input_tensor: Tensor,
    target_tensor: Tensor,
    encoder: EncoderRNN,
    decoder: DecoderRNN,
    encoder_optimizer: Optimizer,
    decoder_optimizer: Optimizer,
    criterion,
):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(-1)
    target_length = target_tensor.size(-1)

    encoder_outputs = torch.zeros(1000, encoder.hidden_size, device=DEVICE)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[:, ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.full(
        (len(target_tensor),), fill_value=SOS_TOKEN, dtype=torch.long, device=DEVICE
    )

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[:, di])
            decoder_input = target_tensor[:, di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[:, di])
            if decoder_input.item() == EOS_TOKEN:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def trainIters(
    dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: DecoderRNN,
    print_every: int = 100,
    plot_every: int = 500,
    learning_rate: float = 0.01,
):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # for iter in range(1, n_iters + 1):
    for i, (input_tensor, target_tensor) in enumerate(dataloader, 1):
        loss = train(
            input_tensor,
            target_tensor,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += loss
        plot_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                "%s (%d %d%%) %.4f"
                % (
                    timeSince(start, i / len(dataloader)),
                    i,
                    i / len(dataloader) * 100,
                    print_loss_avg,
                )
            )

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def collate_fn(input_batch):
    max_ingredients_length = max((len(ingredients) for ingredients, _ in input_batch))
    max_recipe_length = max((len(recepe) for _, recepe in input_batch))

    ingredients_batch = torch.full(
        (len(input_batch), max_ingredients_length),
        fill_value=EOS_TOKEN,
        dtype=torch.long,
        device=DEVICE,
    )
    recipes_batch = torch.full(
        (len(input_batch), max_recipe_length),
        fill_value=EOS_TOKEN,
        dtype=torch.long,
        device=DEVICE,
    )

    for i, (ingredients, recipe) in enumerate(input_batch):
        ingredients_batch[i, : len(ingredients)] = ingredients
        recipes_batch[i, : len(recipe)] = recipe

    return ingredients_batch, recipes_batch


dataset = CookingDataset("data/train")
dataloader = DataLoader(dataset, 64, shuffle=True, collate_fn=collate_fn)

hidden_size = 256
encoder1 = EncoderRNN(dataset.lang.n_words, hidden_size).to(DEVICE)
attn_decoder1 = DecoderRNN(hidden_size, dataset.lang.n_words).to(DEVICE)

trainIters(dataloader, encoder1, attn_decoder1, print_every=100)
