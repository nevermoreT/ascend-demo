"""
数据集: Multi30k 英→德翻译 (通过 HuggingFace datasets 加载)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

PAD, SOS, EOS, UNK = 0, 1, 2, 3


def tokenize(sentence):
    return sentence.strip().lower().split()


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": PAD, "<sos>": SOS, "<eos>": EOS, "<unk>": UNK}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build(self, sentences, min_freq=2):
        freq = {}
        for s in sentences:
            for w in tokenize(s):
                freq[w] = freq.get(w, 0) + 1
        for w, c in sorted(freq.items()):
            if c >= min_freq and w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w

    def encode(self, sentence, max_len=64):
        tokens = tokenize(sentence)
        ids = [SOS] + [self.word2idx.get(w, UNK) for w in tokens] + [EOS]
        return ids[:max_len]

    def decode(self, ids):
        words = []
        for i in ids:
            if i == EOS:
                break
            if i in (PAD, SOS):
                continue
            words.append(self.idx2word.get(i, "<unk>"))
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


class TranslationDataset(Dataset):
    def __init__(self, hf_split, src_vocab, tgt_vocab, max_len=64):
        self.data = hf_split
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids = self.src_vocab.encode(self.data[idx]["en"], self.max_len)
        tgt_ids = self.tgt_vocab.encode(self.data[idx]["de"], self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD)
    tgt_input = tgt_padded[:, :-1]
    tgt_output = tgt_padded[:, 1:]
    return src_padded, tgt_input, tgt_output


def load_data(batch_size=64, max_len=64):
    dataset = load_dataset("bentrevett/multi30k")

    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build([s["en"] for s in dataset["train"]], min_freq=2)
    tgt_vocab.build([s["de"] for s in dataset["train"]], min_freq=2)

    train_ds = TranslationDataset(dataset["train"], src_vocab, tgt_vocab, max_len)
    val_ds = TranslationDataset(dataset["validation"], src_vocab, tgt_vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    return train_loader, val_loader, src_vocab, tgt_vocab


if __name__ == "__main__":
    train_loader, val_loader, src_vocab, tgt_vocab = load_data(batch_size=4)
    src, tgt_in, tgt_out = next(iter(train_loader))
    print(f"词表大小: src={len(src_vocab)}, tgt={len(tgt_vocab)}")
    print(f"训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
    print(f"src shape: {src.shape}, tgt_in shape: {tgt_in.shape}, tgt_out shape: {tgt_out.shape}")
    print(f"src 示例: {src_vocab.decode(src[0].tolist())}")
    print(f"tgt 示例: {tgt_vocab.decode(tgt_out[0].tolist())}")
