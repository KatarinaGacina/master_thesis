import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer_encoder.encoder_block import EncoderN
from modules.hyena.hyena_block import HyenaBlockMR, HyenaBlockLI

from modules.downstream.chromatin_output_head import ChromatinClassificationHead
from modules.downstream.gene_expression_output_head import GeneClassificationHead

from modules.longcontext_adaptation.unet import DownsizeBlock, UpsizeBlock


class CoreBlock(nn.Module):
    def __init__(self, feature_dim, num_enc, num_heads):
        super().__init__()

        self.hyena_layer = HyenaBlockMR(feature_dim, inner_factor=1, filter_len=127, modulate=True)
        self.encoder_layer = EncoderN(num_enc, num_heads, feature_dim)

    def forward(self, x, attn_mask=None):
        out = self.hyena_layer(x)
        out = self.encoder_layer(out, attn_mask)

        return out

class TransformerOnly(nn.Module):
    def __init__(self, feature_dim, num_enc, num_heads):
        super().__init__()

        self.encoder_layer = EncoderN(num_enc, num_heads, feature_dim)

    def forward(self, x, attn_mask=None):
        out = self.encoder_layer(x, attn_mask)

        return out

class HyenaOnly(nn.Module):
    def __init__(self, feature_dim, l_max):
        super().__init__()

        self.hyena_layer_mr = HyenaBlockMR(feature_dim, inner_factor=1, filter_len=127, modulate=True)
        
        self.hyena_layer1 = HyenaBlockLI(feature_dim, l_max, is_causal=False, modulate=True)
        self.hyena_layer2 = HyenaBlockLI(feature_dim, l_max, is_causal=False, modulate=True)

        self.post_norm = nn.RMSNorm(feature_dim, eps=1e-8)

    def forward(self, x, attn_mask=None): #Hyena does not use attention mask
        out = self.hyena_layer_mr(x)

        out = self.hyena_layer1(out)
        out = self.hyena_layer2(out)

        out = self.post_norm(out)

        return out



class DNAModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_layer = nn.Embedding(num_embeddings=config["vocab_size"], embedding_dim=config["embed_dim"], padding_idx=config["pad_index"])
        self.embed_proj = nn.Sequential(
            nn.Conv1d(
                config["embed_dim"],
                config["feature_dim"],
                stride=1,
                kernel_size=config["embed_conv_kernel"],
                padding= config["embed_conv_kernel"] // 2,
                bias=False,
            ),
            nn.GELU()
        )
        #self.embed_proj = nn.Linear(config["embed_dim"], config["feature_dim"], bias=False)

        self.core_block = CoreBlock(config["feature_dim"], config["encoder_num"], config["num_heads"])
        #self.core_block = HyenaOnly(config["feature_dim"], config["outputlen"])
        #self.core_block = TransformerOnly(config["feature_dim"], config["encoder_num"], config["num_heads"])

    def forward(self, x, attn_mask=None):
        out = self.embed_layer(x)

        out = out.transpose(1, 2)
        out = self.embed_proj(out)
        out = out.transpose(1, 2)

        out = self.core_block(out, attn_mask)
        
        return out


class ChromatinModel(nn.Module):
    def __init__(self, config, pretrained=None):
        super().__init__()

        self.dna_model = DNAModel(config)

        if pretrained is not None:
            device = next(self.dna_model.parameters()).device 
            checkpoint_base = torch.load(pretrained, map_location=device, weights_only=True)
            base_weights_dict = checkpoint_base.get('base_model_weights')

            if base_weights_dict is None:
                print("Randomly initialized model.")
            else:
                try:
                    self.dna_model.load_state_dict(base_weights_dict)
                    print("Loaded pretrained model weights.")
                except RuntimeError as e:
                    print(f"Loading failed due to mismatch. Randomly initialized model.")
        else:
            print("Randomly initialized model.")

        self.class_head = ChromatinClassificationHead(feature_dim=(config["feature_dim"]))

    def forward(self, x, attn_mask=None):
        out = self.dna_model(x, attn_mask)
        out = self.class_head(out)
        
        return out


class DNAModelLongContext(nn.Module):
    def __init__(self, config, layer_num=7):
        super().__init__()

        self.embed_layer = nn.Embedding(num_embeddings=config["vocab_size"], embedding_dim=config["embed_dim"], padding_idx=config["pad_index"])
        self.conv_embed = nn.Sequential(
            nn.Conv1d(
                config["embed_dim"],
                config["feature_dim"],
                stride=1,
                kernel_size=config["embed_conv_kernel"],
                padding= config["embed_conv_kernel"] // 2,
                bias=False,
            ),
            nn.ReLU()
        )

        self.layer_num = layer_num

        self.downsize_block = DownsizeBlock(layer_num, config["feature_dim"])
        intermediate_embed_value = config["feature_dim"] + layer_num*self.downsize_block.get_embedding_factor()

        self.core_block = CoreBlock(intermediate_embed_value, config["encoder_num"], config["num_heads"])
        self.upsize_block = UpsizeBlock(layer_num, intermediate_embed_value)

    def forward(self, x, attn_mask=None):
        if attn_mask is not None:
            mask = F.max_pool1d(attn_mask[:, 0, 0, :].float(), kernel_size=(2**self.layer_num), stride=(2**self.layer_num))[:, None, None, :].bool()
        else:
            mask = None

        out = self.embed_layer(x)

        out = out.transpose(1,2)
        out = self.conv_embed(out)
        
        out, r = self.downsize_block(out)
        assert len(r) > 0
        out = out.transpose(1,2)

        out = self.core_block(out, mask)

        out = out.transpose(1,2)
        out = self.upsize_block(out, r)
        out = out.transpose(1,2)
        
        return out

class ChromatinModelLongContext(nn.Module):
    def __init__(self, config, pretrained=None):
        super().__init__()

        self.dna_model = DNAModelLongContext(config)

        if pretrained is not None:
            device = next(self.dna_model.parameters()).device 
            checkpoint_base = torch.load(pretrained, map_location=device, weights_only=True)
            base_weights_dict = checkpoint_base.get('base_model_weights')

            if base_weights_dict is None:
                print("Randomly initialized model.")
            else:
                try:
                    self.dna_model.load_state_dict(base_weights_dict)
                    print("Loaded pretrained model weights.")
                except RuntimeError as e:
                    print(f"Loading failed due to mismatch. Randomly initialized model.")
        else:
            print("Randomly initialized model.")

        self.class_head = ChromatinClassificationHead(feature_dim=(config["feature_dim"]))

    def forward(self, x, attn_mask=None):
        out = self.dna_model(x, attn_mask)
        out = self.class_head(out)
        
        return out
    

class GeneExpressionModelLongContext(nn.Module):
    def __init__(self, config, pretrained=None):
        super().__init__()

        self.dna_model = DNAModelLongContext(config, layer_num=7)

        if pretrained is not None:
            device = next(self.dna_model.parameters()).device 
            checkpoint_base = torch.load(pretrained, map_location=device, weights_only=True)
            base_weights_dict = checkpoint_base.get('base_model_weights')

            if base_weights_dict is None:
                print("Randomly initialized model.")
            else:
                try:
                    self.dna_model.load_state_dict(base_weights_dict)
                    print("Loaded pretrained model weights.")
                except RuntimeError as e:
                    print(f"Loading failed due to mismatch. Randomly initialized model.")
        else:
            print("Randomly initialized model.")

        self.class_head = GeneClassificationHead(feature_dim=(config["feature_dim"]))

    def forward(self, x, attn_mask=None):
        out = self.dna_model(x, attn_mask)
        out = self.class_head(out)
        
        return out
    

from modules.downstream.task_head import TaskClassificationHead

class FinetuneModel(nn.Module):
    def __init__(self, config, pretrained=None):
        super().__init__()

        self.dna_model = DNAModel(config)

        if pretrained is not None:
            device = next(self.dna_model.parameters()).device 
            checkpoint_base = torch.load(pretrained, map_location=device, weights_only=True)
            base_weights_dict = checkpoint_base.get('base_model_weights')

            if base_weights_dict is None:
                print("Randomly initialized model.")
            else:
                try:
                    self.dna_model.load_state_dict(base_weights_dict)
                    print("Loaded pretrained model weights.")
                except RuntimeError as e:
                    print(f"Loading failed due to mismatch. Randomly initialized model.")
        else:
            print("Randomly initialized model.")

        self.task_head = TaskClassificationHead(feature_dim=config["feature_dim"], output_dim=config["number_labels"])

    def forward(self, x, attn_mask=None):
        out = self.dna_model(x, attn_mask)
        out = self.task_head(out)
        
        return out
    
class FinetuneLongModel(nn.Module):
    def __init__(self, config, pretrained=None):
        super().__init__()

        self.dna_model = DNAModelLongContext(config, layer_num=7)

        if pretrained is not None:
            device = next(self.dna_model.parameters()).device 
            checkpoint_base = torch.load(pretrained, map_location=device, weights_only=True)
            base_weights_dict = checkpoint_base.get('base_model_weights')

            if base_weights_dict is None:
                print("Randomly initialized model.")
            else:
                try:
                    self.dna_model.load_state_dict(base_weights_dict)
                    print("Loaded pretrained model weights.")
                except RuntimeError as e:
                    print(f"Loading failed due to mismatch. Randomly initialized model.")
        else:
            print("Randomly initialized model.")

        self.task_head = TaskClassificationHead(feature_dim=config["feature_dim"], output_dim=config["number_labels"])

    def forward(self, x, attn_mask=None):
        out = self.dna_model(x, attn_mask)
        out = self.task_head(out)
        
        return out