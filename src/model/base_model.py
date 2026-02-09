import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import DNAModel, DNAModelLongContext


class MaskHead(nn.Module):
    def __init__(self, feature_dim, vocab_size):
        super().__init__()

        self.linear1 = nn.Linear(feature_dim, feature_dim)
        self.rms_norm = nn.RMSNorm(feature_dim)
        self.linear2 = nn.Linear(feature_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.rms_norm(x)
        x = self.linear2(x)

        return x


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dna_model = DNAModel(config)

        vocab_size = config.get("output_vocab_size", config["vocab_size"])
        self.mask_head = MaskHead(feature_dim=config["feature_dim"], vocab_size=vocab_size)
    
    def forward(self, x, attn_mask=None):
        out = self.dna_model(x, attn_mask)
        logits = self.mask_head(out)

        result = {"logits": logits, "representations": out}
        return result


class BaseModelLongContext(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dna_model = DNAModelLongContext(config)

        vocab_size = config.get("output_vocab_size", config["vocab_size"])
        self.mask_head = MaskHead(feature_dim=config["feature_dim"], vocab_size=vocab_size)
    
    def forward(self, x, attn_mask=None):
        out = self.dna_model(x, attn_mask)
        logits = self.mask_head(out)

        result = {"logits": logits, "representations": out}
        return result