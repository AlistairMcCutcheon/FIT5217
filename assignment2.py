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
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
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
import nltk

nltk.download("wordnet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN = 0
EOS_TOKEN = 1
UNK_TOKEN = 2


class CookingDataset(Dataset):
    def __init__(
        self, data_dir_path: str, lang=None, max_len: int = 150, preprocess_data=False
    ) -> None:
        super().__init__()

        self.ingredients, self.recipes = read_ingredients_recepies(
            data_dir_path, preprocess_data=preprocess_data
        )

        self.lang = (
            create_lang(self.ingredients, self.recipes) if lang is None else lang
        )

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
            if len(x) <= max_len and len(y) <= max_len
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
        return len(self.tokenised_ingredients)

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

        print("Vocabulary Size, including SOS_TOKEN, EOS_TOKEN, and UNK_TOKEN:")
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
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS
        self.addWord("SOS")
        self.addWord("EOS")
        self.addWord("UNK")

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

    def get_index(self, word: str):
        try:
            return self.word2index[word]
        except KeyError:
            return self.word2index["UNK"]

    def create_index_to_prob(self):
        index_to_count = torch.zeros((self.n_words,), requires_grad=False)
        for index, word in self.index2word.items():
            index_to_count[index] = self.word2count[word]
        return index_to_count / torch.sum(index_to_count)


# class EncoderRNN(nn.Module):
#     def __init__(self, hidden_size, layers=1):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.gru = nn.GRU(hidden_size, hidden_size, num_layers=layers)

#     def forward(self, x: PackedSequence):
#         output, hidden = self.gru(x)
#         return output, hidden


# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, layers=1):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.gru = nn.GRU(hidden_size, hidden_size, num_layers=layers)

#     def forward(self, target: PackedSequence, hidden):
#         return self.gru(target, hidden)


class Embedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: PackedSequence):
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        x = self.embedding(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        return x


class Head(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.out = nn.Linear(input_size, output_size)

    def forward(self, x: PackedSequence):
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        x = self.out(x)
        x = F.log_softmax(x, dim=2)
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        return x


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn: nn.Module, network: nn.Module):
        super(AttnDecoderRNN, self).__init__()
        self.network = network
        self.attn = attn

    def forward(self, x: PackedSequence, hidden, encoder_outputs: PackedSequence):
        decoder_output, hidden = self.network(x, hidden)

        attn_out, attn_dist = self.attn(
            key=encoder_outputs, value=encoder_outputs, query=decoder_output
        )
        return attn_out, decoder_output, attn_dist, hidden


class PointerGen(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.context_linear = nn.Linear(hidden_size, 1)
        self.decoder_out_linear = nn.Linear(hidden_size, 1)
        self.decoder_input_linear = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_tokens: PackedSequence,
        context: PackedSequence,
        decoder_input: PackedSequence,
        decoder_output: PackedSequence,
        vocab_dist: PackedSequence,
        attn_dist: PackedSequence,
        encoder_outputs: PackedSequence,
    ):
        input_tokens, _ = pad_packed_sequence(input_tokens, batch_first=True)
        decoder_input, _ = pad_packed_sequence(decoder_input, batch_first=True)
        decoder_output, _ = pad_packed_sequence(decoder_output, batch_first=True)
        context, _ = pad_packed_sequence(context, batch_first=True)
        vocab_dist, lengths = pad_packed_sequence(vocab_dist, batch_first=True)
        attn_dist, _ = pad_packed_sequence(attn_dist, batch_first=True)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)

        p_gen = F.sigmoid(
            self.context_linear(context)
            + self.decoder_out_linear(decoder_output)
            + self.decoder_input_linear(decoder_input)
        )
        copy_dist = torch.scatter_add(
            input=torch.zeros_like(vocab_dist),
            dim=2,
            index=input_tokens.unsqueeze(1).expand(attn_dist.shape),
            src=attn_dist,
        )
        final_dist = vocab_dist * p_gen + (1 - p_gen) * F.log_softmax(copy_dist, dim=2)
        return pack_padded_sequence(
            final_dist, lengths, batch_first=True, enforce_sorted=False
        )


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
                    sequence[: input_seq_length - 1],
                    target_sequence[1:input_seq_length],
                    reduction="sum",
                )
                / input_seq_length
            )
        return loss / len(input_seq_unpacked)


class DotProductAttention(nn.Module):
    def forward(
        self, key: PackedSequence, value: PackedSequence, query: PackedSequence
    ):
        query, lengths = pad_packed_sequence(query, batch_first=True)
        key, _ = pad_packed_sequence(key, batch_first=True)
        value, _ = pad_packed_sequence(value, batch_first=True)

        attn_scores = torch.bmm(query, key.transpose(2, 1))
        attn_dist = F.softmax(attn_scores, dim=2)
        attn_out = torch.bmm(attn_dist, value)

        attn_out = pack_padded_sequence(
            attn_out, lengths, batch_first=True, enforce_sorted=False
        )
        attn_dist = pack_padded_sequence(
            attn_dist, lengths, batch_first=True, enforce_sorted=False
        )
        return attn_out, attn_dist


class BahdanauAttention(nn.Module):
    def __init__(self, query_size) -> None:
        super().__init__()
        self.query_size = query_size

        self.key_linear = nn.Linear(query_size, query_size)
        self.query_linear = nn.Linear(query_size, query_size)

        self.v_linear = nn.Linear(query_size, 1)

    def forward(
        self, key: PackedSequence, value: PackedSequence, query: PackedSequence
    ):
        query, lengths = pad_packed_sequence(query, batch_first=True)
        key, _ = pad_packed_sequence(key, batch_first=True)
        value, _ = pad_packed_sequence(value, batch_first=True)

        embedded_key = self.key_linear(key)
        embedded_query = self.query_linear(query)

        x = F.tanh(embedded_key.unsqueeze(1) + embedded_query.unsqueeze(2))
        attn_scores = self.v_linear(x.view(-1, self.query_size)).view(*x.shape[:3])

        attn_dist = F.softmax(attn_scores, 2)
        attn_out = torch.bmm(attn_dist, value)

        attn_out = pack_padded_sequence(
            attn_out, lengths, batch_first=True, enforce_sorted=False
        )
        attn_dist = pack_padded_sequence(
            attn_dist, lengths, batch_first=True, enforce_sorted=False
        )
        return attn_out, attn_dist


class SeqToSeq(nn.Module):
    def __init__(
        self,
        encoder: nn.LSTM,
        decoder: nn.LSTM,
        encoder_embedding: Embedding,
        decoder_embedding: Embedding,
        head: Head,
        max_len=150,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_embedding = encoder_embedding
        self.decoder_embedding = decoder_embedding
        self.head = head
        self.max_len = max_len

    def forward(self, x: PackedSequence, target: PackedSequence):
        _, hidden = self.forward_encoder(x)

        if len(target.batch_sizes) > 1:
            decoder_output, _ = self.forward_decoder_parallel(target, hidden)
            return self.head(decoder_output)

        return self.forward_decoder_sequential(target, hidden)

    def forward_encoder(self, x: PackedSequence):
        return self.encoder(self.encoder_embedding(x))

    def forward_decoder_parallel(self, x: PackedSequence, hidden: PackedSequence):
        return self.decoder(self.decoder_embedding(x), hidden)

    def forward_decoder_sequential(self, x, hidden):
        outputs = torch.zeros((x.batch_sizes[0], self.max_len), device=DEVICE)
        for timestep in range(self.max_len):
            decoder_output, hidden = self.forward_decoder_parallel(x, hidden)
            x = self.head(decoder_output)
            x, lengths = pad_packed_sequence(x, batch_first=True)
            x = torch.argmax(x, dim=2)
            outputs[:, timestep] = x.squeeze(1)  # in place operation
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        outputs = torch.cat(
            (outputs, torch.full((outputs.shape[0], 1), EOS_TOKEN, device=DEVICE)),
            dim=1,
        )
        lengths = torch.argmax(torch.eq(outputs, EOS_TOKEN).to(torch.uint8), dim=1) + 1
        return pack_padded_sequence(
            outputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )


class SeqToSeqWithAttention(nn.Module):
    def __init__(
        self,
        encoder: nn.LSTM,
        decoder: nn.LSTM,
        attn: nn.Module,
        encoder_embedding: Embedding,
        decoder_embedding: Embedding,
        head: Head,
        max_len=150,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attn = attn
        self.encoder_embedding = encoder_embedding
        self.decoder_embedding = decoder_embedding
        self.head = head
        self.max_len = max_len

    def forward(self, x, target):
        encoder_outputs, hidden = self.forward_encoder(x)

        if len(target.batch_sizes) > 1:
            attn_out, decoder_outputs, _, _ = self.forward_decoder_parallel(
                target, hidden, encoder_outputs
            )
            features = concat_packed_sequences(attn_out, decoder_outputs)

            return self.head(features)

        return self.forward_decoder_sequential(target, hidden, encoder_outputs)

    def forward_encoder(self, x: PackedSequence):
        return self.encoder(self.encoder_embedding(x))

    def forward_decoder_parallel(
        self, x: PackedSequence, hidden: PackedSequence, encoder_outputs: PackedSequence
    ):
        x = self.decoder_embedding(x)
        decoder_output, hidden = self.decoder(x, hidden)

        attn_out, attn_dist = self.attn(
            key=encoder_outputs, value=encoder_outputs, query=decoder_output
        )
        return attn_out, decoder_output, attn_dist, hidden

    def forward_decoder_sequential(self, x, hidden, encoder_outputs):
        outputs = torch.zeros((x.batch_sizes[0], self.max_len), device=DEVICE)
        for timestep in range(self.max_len):
            attn_out, decoder_outputs, _, hidden = self.forward_decoder_parallel(
                x, hidden, encoder_outputs
            )
            features = concat_packed_sequences(attn_out, decoder_outputs)
            x = self.head(features)
            x, lengths = pad_packed_sequence(x, batch_first=True)
            x = torch.argmax(x, dim=2)
            outputs[:, timestep] = x.squeeze(1)  # in place operation
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        outputs = torch.cat(
            (outputs, torch.full((outputs.shape[0], 1), EOS_TOKEN, device=DEVICE)),
            dim=1,
        )
        lengths = torch.argmax(torch.eq(outputs, EOS_TOKEN).to(torch.uint8), dim=1) + 1
        return pack_padded_sequence(
            outputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )


class GetToThePoint(nn.Module):
    def __init__(
        self,
        encoder: nn.LSTM,
        decoder: nn.LSTM,
        attn: nn.Module,
        encoder_embedding: Embedding,
        decoder_embedding: Embedding,
        head: Head,
        pointer_gen: PointerGen,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attn = attn
        self.encoder_embedding = encoder_embedding
        self.decoder_embedding = decoder_embedding
        self.head = head
        self.pointer_gen = pointer_gen
        self.max_len = 150

    def forward(self, x, target):
        encoder_outputs, encoder_hidden = self.forward_encoder(x)
        if len(target.batch_sizes) > 1:
            # print(pad_packed_sequence(x, batch_first=True)[0])
            # print(pad_packed_sequence(target, batch_first=True)[0])
            target = self.decoder_embedding(target)
            # attn_out, decoder_output, attn_dist, _ = self.decoder(
            #     target, encoder_hidden, encoder_outputs
            # )

            decoder_output, _ = self.decoder(target, encoder_hidden)
            attn_out, attn_dist = self.attn(
                key=encoder_outputs, value=encoder_outputs, query=decoder_output
            )

            features = concat_packed_sequences(attn_out, decoder_output)
            vocab_dist = self.head(features)
            # print(pad_packed_sequence(vocab_dist, batch_first=True)[0].shape)
            final_dist = self.pointer_gen(
                input_tokens=x,
                context=attn_out,
                decoder_input=target,
                decoder_output=decoder_output,
                vocab_dist=vocab_dist,
                attn_dist=attn_dist,
                encoder_outputs=encoder_outputs,
            )
            final_dist = vocab_dist
            return final_dist
        return self.forward_decoder_sequential(
            x, target, encoder_hidden, encoder_outputs
        )

    def forward_decoder_sequential(self, x, target, hidden, encoder_outputs):
        outputs = torch.zeros((target.batch_sizes[0], self.max_len), device=DEVICE)
        for timestep in range(self.max_len):
            target = self.decoder_embedding(target)
            # (
            #     attn_out,
            #     decoder_outputs,
            #     attn_dist,
            #     hidden,
            # ) = self.decoder(target, hidden, encoder_outputs)

            decoder_outputs, hidden = self.decoder(target, hidden)
            attn_out, attn_dist = self.attn(
                key=encoder_outputs, value=encoder_outputs, query=decoder_outputs
            )

            features = concat_packed_sequences(attn_out, decoder_outputs)
            vocab_dist = self.head(features)
            final_dist = self.pointer_gen(
                input_tokens=x,
                context=attn_out,
                decoder_input=target,
                decoder_output=decoder_outputs,
                vocab_dist=vocab_dist,
                attn_dist=attn_dist,
                encoder_outputs=encoder_outputs,
            )

            final_dist, lengths = pad_packed_sequence(final_dist, batch_first=True)
            target = torch.argmax(final_dist, dim=2)

            outputs[:, timestep] = target.squeeze(1)

            target = pack_padded_sequence(
                target, lengths, batch_first=True, enforce_sorted=False
            )
        outputs = torch.cat(
            (outputs, torch.full((outputs.shape[0], 1), EOS_TOKEN, device=DEVICE)),
            dim=1,
        )
        lengths = torch.argmax(torch.eq(outputs, EOS_TOKEN).to(torch.uint8), dim=1) + 1
        return pack_padded_sequence(
            outputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

    def forward_encoder(self, x: PackedSequence):
        return self.encoder(self.encoder_embedding(x))


def concat_packed_sequences(seq1: PackedSequence, seq2: PackedSequence):
    seq1, _ = pad_packed_sequence(seq1, batch_first=True)
    seq2, lens = pad_packed_sequence(seq2, batch_first=True)
    out = torch.cat((seq1, seq2), dim=2)
    return pack_padded_sequence(out, lens, batch_first=True, enforce_sorted=False)


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
    indexes = [lang.get_index(word) for word in text.split(" ")]
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


def read_ingredients_recepies(data_dir_path: str, preprocess_data=False):
    ingredients_per_recepie = []
    recepies = []
    i = 0
    for recipes_file_path in Path(data_dir_path).glob("*"):
        with open(recipes_file_path, "r") as file:
            for ingredients, recepie in read_ingredient_recipe(file):
                if preprocess_data:
                    ingredients_per_recepie.append(normalise_string(ingredients))
                    recepies.append(normalise_string(recepie))
                    continue
                ingredients_per_recepie.append(ingredients)
                recepies.append(recepie)
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
    train_dataloader: DataLoader,
    seq_to_seq,
    optimiser: Optimizer,
    criterion,
    writer: SummaryWriter,
    lang: Lang,
    epochs: int = 10,
    test_dataloader: DataLoader = None,
):
    for epoch in range(epochs):
        for i, (input_tensor, target_tensor) in enumerate(train_dataloader):
            seq_to_seq.zero_grad()

            output = seq_to_seq.forward(input_tensor, target_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()

            optimiser.step()

            iteration = epoch * len(train_dataloader) + i
            writer.add_scalars("Loss", {"Train": loss.item()}, iteration)

            if test_dataloader is None:
                continue
            if i % 100 == 0:
                dev_loss, dev_bleu, dev_meteor = eval(
                    dev_dataloader,
                    seq_to_seq,
                    criterion,
                    lang,
                )

                writer.add_scalars("Loss", {"Dev": dev_loss}, iteration)
                writer.add_scalars("Bleu", {"Dev": dev_bleu}, iteration)
                writer.add_scalars("Meteor", {"Dev": dev_meteor}, iteration)


def eval(
    dataloader: DataLoader,
    seq_to_seq: nn.Module,
    criterion: nn.Module,
    lang: Lang,
):
    total_loss = 0
    total_bleu = 0
    total_meteor = 0
    for input_tensor, target_tensor in dataloader:
        with torch.no_grad():
            parallel_output = seq_to_seq.forward(input_tensor, target_tensor)
            total_loss += criterion(parallel_output, target_tensor).item()
            sequential_output = seq_to_seq.forward(
                input_tensor,
                pack_sequence(
                    torch.full(
                        (input_tensor.batch_sizes[0], 1), SOS_TOKEN, device=DEVICE
                    )
                ),
            )
        output, output_lengths = pad_packed_sequence(
            sequential_output, batch_first=True
        )
        target_tensor, target_lengths = pad_packed_sequence(
            target_tensor, batch_first=True
        )
        for target_text, target_length, output_text, output_length in zip(
            target_tensor, target_lengths, output, output_lengths
        ):
            target_text = target_text[1:target_length]
            output_text = output_text[:output_length]
            total_bleu += sentence_bleu(
                [target_text.tolist()], output_text.tolist()
            ) / len(output)
            total_meteor += meteor_score(
                [tokens_to_words(target_text, lang)],
                tokens_to_words(output_text, lang),
            ) / len(output)

    return (
        total_loss / len(dataloader),
        total_bleu / len(dataloader),
        total_meteor / len(dataloader),
    )
    # test_text = "10 oz chopped broccoli, 2 tbsp butter, 2 tbsp flour, 1/2 tsp salt, 1/4 tsp black pepper, 1/4 tsp ground nutmeg, 1 cup milk, 1 1/2 cup shredded swiss cheese, 2 tsp lemon juice, 2 cup cooked cubed turkey, 4 oz mushrooms, 1/4 cup grated Parmesan cheese, 1 can refrigerated biscuits"
    # input = pack_sequence(
    #     [
    #         torch.tensor(
    #             indexes_from_text(
    #                 train_dataset.lang,
    #                 test_text,
    #                 sos_token=False,
    #                 eos_token=True,
    #             ),
    #             dtype=torch.long,
    #             device=DEVICE,
    #         )
    #     ]
    # )
    # print(inference(input, seq_to_seq, lang))


def inference(input: PackedSequence, model: nn.Module, lang: Lang):
    target = pack_sequence(
        torch.full((input.batch_sizes[0], 1), SOS_TOKEN, device=DEVICE)
    )
    with torch.no_grad():
        # print(target)
        batch_tokens = model.forward(input, target)

    return batch_tokens_to_words(batch_tokens, lang)


def batch_tokens_to_words(batch_tokens: Tensor, lang: Lang):
    batch_words = []
    for tokens in batch_tokens:
        batch_words.append(
            tokens_to_words(tokens, lang)
        )  # [lang.index2word[token.item()] for token in tokens])
    return batch_words


def tokens_to_words(tokens: Tensor, lang: Lang):
    return [lang.index2word[token.item()] for token in tokens]


def collate_fn(input_batch):
    ingredients_batch = pack_sequence(
        [ingredients for ingredients, _ in input_batch], enforce_sorted=False
    )
    recipies_batch = pack_sequence(
        [recipe for _, recipe in input_batch], enforce_sorted=False
    )

    return ingredients_batch, recipies_batch


train_dataset = CookingDataset("data/train", max_len=150)

dev_dataset = CookingDataset("data/dev", lang=train_dataset.lang, max_len=150)

print("")
print("")
print("TRAINING SET")
print("")
print("")
train_dataset.summarise()
print("")
print("")
print("DEV SET")
print("")
print("")
dev_dataset.summarise()
train_dataloader = DataLoader(
    train_dataset, 32, shuffle=True, collate_fn=collate_fn, drop_last=True
)
print(f"Length of train dataloader: {len(train_dataloader)}")
dev_dataloader = DataLoader(dev_dataset, 32, shuffle=True, collate_fn=collate_fn)
print(f"Length of dev dataloader: {len(dev_dataloader)}")

hidden_size = 256

encoder_embedding = Embedding(train_dataset.lang.n_words, hidden_size).to(DEVICE)
decoder_embedding = Embedding(train_dataset.lang.n_words, hidden_size).to(DEVICE)
encoder = nn.LSTM(hidden_size, hidden_size, 1).to(DEVICE)
decoder = nn.LSTM(hidden_size, hidden_size, 1).to(DEVICE)
# head = Head(hidden_size, train_dataset.lang.n_words)
# seq_to_seq = SeqToSeq(encoder, decoder, encoder_embedding, decoder_embedding, head).to(
#     DEVICE
# )

# decoder = AttnDecoderRNN(hidden_size, BahdanauAttention(256)).to(
#     DEVICE
# )
# decoder = AttnDecoderRNN(hidden_size, DotProductAttention()).to(DEVICE)
attn = DotProductAttention().to(DEVICE)
attn = BahdanauAttention(hidden_size).to(DEVICE)

# decoder = AttnDecoderRNN(hidden_size, BahdanauAttention(256)).to(DEVICE)
head = Head(2 * hidden_size, train_dataset.lang.n_words)
# seq_to_seq = SeqToSeqWithAttention(
#     encoder, decoder, attn, encoder_embedding, decoder_embedding, head
# ).to(DEVICE)

pointer_gen = PointerGen(hidden_size).to(DEVICE)
seq_to_seq = GetToThePoint(
    encoder, decoder, attn, encoder_embedding, decoder_embedding, head, pointer_gen
).to(DEVICE)

writer = SummaryWriter()

learning_rate = 0.001
optimiser = optim.Adam(seq_to_seq.parameters(), lr=learning_rate)
criterion = SequenceNLLLoss()

train(
    train_dataloader,
    seq_to_seq,
    optimiser,
    criterion,
    writer,
    train_dataset.lang,
    epochs=500,
    test_dataloader=dev_dataloader,
)
# test_text = "10 oz chopped broccoli, 2 tbsp butter, 2 tbsp flour, 1/2 tsp salt, 1/4 tsp black pepper, 1/4 tsp ground nutmeg, 1 cup milk, 1 1/2 cup shredded swiss cheese, 2 tsp lemon juice, 2 cup cooked cubed turkey, 4 oz mushrooms, 1/4 cup grated Parmesan cheese, 1 can refrigerated biscuits"
# input = pack_sequence(
#     [
#         torch.tensor(
#             indexes_from_text(
#                 train_dataset.lang,
#                 test_text,
#                 sos_token=False,
#                 eos_token=True,
#             ),
#             dtype=torch.long,
#             device=DEVICE,
#         )
#     ]
# )
# inference(input, seq_to_seq, lang)
