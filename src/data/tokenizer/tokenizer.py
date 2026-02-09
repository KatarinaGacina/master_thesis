import torch
from transformers import PreTrainedTokenizer

from itertools import chain
import numpy as np

class DNATokenizer:
    def __init__(self, chromatin_tokens=False):
        
        self.tokens = ["A", "T", "C", "G", "m", "h"]
        self.special_tokens = ["N", "<PAD>", "<CLS>", "<MASK>", "<END>"]

        if chromatin_tokens:
            self.tokens.extend(["a", "t", "c", "g", "M", "H"])
            self.special_tokens.append("n")

        self.itos = {i: t for i, t in enumerate(self.tokens + self.special_tokens)}
        self.stoi = {t: i for i, t in self.itos.items()}

        self.special_tokens_ids = []
        for s in self.special_tokens:
            self.special_tokens_ids.append(self.stoi[s])

        self.pad_token_id = self.stoi["<PAD>"]
        self.mask_token_id = self.stoi["<MASK>"]
        self.unk_token_id = self.stoi["N"]

    def encode(self, sequence, include_cls=False):
        lookup = self.stoi
        
        cls_id = lookup["<CLS>"]
        unk_id = lookup["N"]

        seq_codes = np.fromiter(
            (lookup.get(c, unk_id) for c in sequence),
            dtype=np.int64,
            count=len(sequence)
        )

        if include_cls:
            encoded = np.empty(len(sequence) + 1, dtype=np.int64)
            encoded[0] = cls_id
            encoded[1:] = seq_codes
        else:
            encoded = np.array(seq_codes, dtype=np.int64)

        return torch.from_numpy(encoded)
    
    def get_pad_index(self):
        return self.stoi["<PAD>"]

    def get_vocab(self):
        return self.stoi

    @property
    def vocab_size(self):
        return len(self.stoi)
    
    @property
    def all_special_tokens(self):
        return self.special_tokens
    
    @property
    def all_special_ids(self):
        return self.special_tokens_ids

    @property
    def for_collator(self):
        return len(self.stoi) - len(self.special_tokens) - 1 #-1 because of h


class DNATokenizerHF(PreTrainedTokenizer): #used for base, in future these two tokenizers will be one
    def __init__(self, **kwargs):

        self.tokens = ["A", "T", "C", "G", "m", "h"]
        self.special_tokens = ["N", "<PAD>", "<CLS>", "<MASK>", "<END>"]

        self._itos = {i: t for i, t in enumerate(self.tokens + self.special_tokens)}
        self._stoi = {t: i for i, t in self._itos.items()}

        super().__init__(
            pad_token="<PAD>",
            cls_token="<CLS>",
            eos_token="<END>",
            mask_token="<MASK>",
            unk_token="N",
            **kwargs
        )

        self.pad_token_id = self._stoi[self.pad_token]
        self.cls_token_id = self._stoi[self.cls_token]
        self.eos_token_id = self._stoi[self.eos_token]
        self.mask_token_id = self._stoi[self.mask_token]
        self.unk_token_id = self._stoi[self.unk_token]

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self._stoi.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self._itos.get(index, self.unk_token)

    def get_vocab(self):
        return self._stoi

    @property
    def vocab_size(self):
        return len(self._stoi)

    @property
    def all_special_tokens(self):
        return self.special_tokens
