import re
import torch
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

class Lang:
    """Class to hold a language vocabulary mapping string to index and vice versa."""
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": PAD_TOKEN, "<SOS>": SOS_TOKEN, "<EOS>": EOS_TOKEN, "<UNK>": UNK_TOKEN}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "<PAD>", SOS_TOKEN: "<SOS>", EOS_TOKEN: "<EOS>", UNK_TOKEN: "<UNK>"}
        self.n_words = 4  # Count PAD, SOS, EOS, UNK

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def normalize_string(s):
    """
    Lowercases, trims, and separates punctuation.
    Designed to preserve Unicode points (like Hindi).
    """
    s = str(s).lower().strip()
    # Separate punctuations with space
    s = re.sub(r"([.!?])", r" \1", s)
    # Remove bizarre symbols but keep multi-lingual text
    s = re.sub(r"[^\w\s.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s)
    return s.strip()


def indexes_from_sentence(lang, sentence):
    """Returns a list of token indices for a given sentence string using the Lang vocab."""
    return [lang.word2index.get(word, UNK_TOKEN) for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence, device):
    """Returns a tensor of indexes appended with the EOS token."""
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device)


class TranslationDataset(Dataset):
    """
    PyTorch Dataset wrapper for translating language pairs.
    It expects pairs of normalized strings.
    """
    def __init__(self, pairs, input_lang, output_lang, max_len=50):
        self.pairs = [p for p in pairs if len(p[0].split()) < max_len and len(p[1].split()) < max_len]
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_len = max_len
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        src_sentence, trg_sentence = self.pairs[idx]
        
        src_indexes = indexes_from_sentence(self.input_lang, src_sentence)
        trg_indexes = indexes_from_sentence(self.output_lang, trg_sentence)
        
        src_indexes.append(EOS_TOKEN)
        trg_indexes.append(EOS_TOKEN)
        
        return src_indexes, trg_indexes

def pad_sequence_custom(sequences, max_len, pad_value=PAD_TOKEN):
    """Pads a list of sequence arrays to be of equievalent length max_len."""
    padded_seqs = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [pad_value] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded_seqs.append(seq)
    return torch.tensor(padded_seqs, dtype=torch.long)

def collate_fn_pad(batch):
    """Collate function for DataLoader padding batches to the max length sequences in the batch."""
    src_batch, trg_batch = [], []
    for src_item, trg_item in batch:
        src_batch.append(src_item)
        trg_batch.append(trg_item)
        
    # Get max lengths in this batch
    src_max_len = max([len(s) for s in src_batch])
    trg_max_len = max([len(t) for t in trg_batch])
    
    src_padded = pad_sequence_custom(src_batch, src_max_len)
    trg_padded = pad_sequence_custom(trg_batch, trg_max_len)
    
    return src_padded, trg_padded

def get_dataloader(pairs, input_lang, output_lang, batch_size=32, max_len=50):
    dataset = TranslationDataset(pairs, input_lang, output_lang, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_pad)
