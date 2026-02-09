import os

os.environ["NUMBA_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from numba import set_num_threads
set_num_threads(4)

import numpy as np
import torch
from torch.utils.data import DataLoader
import umap
from sklearn.manifold import TSNE

import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.preprocessing import normalize

from pl_models.base import BaseLightningModule
from data.chromatin.dataset import ChromDataset
from data.chromatin.datamodule import pad_collate_fn
from data.tokenizer.tokenizer import DNATokenizerHF
from data.load_utils.data_loading import load_intervals_per_split
from configs.config import get_config_base

from functools import partial


def extract_sequence_embeddings(model, dataloader, device):
    model.eval()

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"][:, None, None, :].to(device)
            class_labels = batch["labels"].to(device)

            embeddings = model(input_ids, attention_mask)["representations"]

            mask = attention_mask[:, 0, 0, :].bool()
            embeddings = embeddings * mask.unsqueeze(-1)

            seq_embeddings = embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            seq_labels = (class_labels.bool() & mask).any(dim=1).long()

            embeddings_list.append(seq_embeddings.cpu())
            labels_list.append(seq_labels.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return embeddings.numpy(), labels.numpy()

def extract_embeddings(model, dataloader, device):
    model.eval()

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"][:, None, None, :].to(device)
            class_labels = batch["labels"].to(device)

            embeddings = model(input_ids, attention_mask)["representations"]

            embeddings_flat = embeddings.reshape(-1, embeddings.shape[-1])
            labels_flat = class_labels.reshape(-1)

            mask = attention_mask[:, 0, 0, :].reshape(-1).bool()

            embeddings_list.append(embeddings_flat[mask].cpu())
            labels_list.append(labels_flat[mask].cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return embeddings.numpy(), labels.numpy()


def visualize_umap_result(embeddings_np, labels_np, name):
    #embeddings_np = normalize(embeddings_np)

    print(len(embeddings_np))

    #n_pca_components = 16
    #pca = PCA(n_components=n_pca_components, random_state=42)
    #embeddings_pca = pca.fit_transform(embeddings_np)

    reducer = umap.UMAP(n_neighbors=50, min_dist=0.4, n_components=2, random_state=42, low_memory=True)
    embedding_2d = reducer.fit_transform(embeddings_np) #, y=labels_np)

    """tsne = TSNE(
        n_components=2,    
        perplexity=10,
        learning_rate=200,
        max_iter=1000,
        random_state=42
    )
    embedding_2d = tsne.fit_transform(embeddings_pca)"""

    print("Finished calculation!")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels_np, cmap='tab10', s=5)
    plt.colorbar(scatter)
    plt.title(f"Visualization {name}")

    plt.savefig(f"/checkpoints/umap_embeddings_{name}.png", dpi=300, bbox_inches='tight')

    plt.close()

def main():

    device = "cuda"
    num_examples = 10

    fasta_name = "/path/assembly.fasta.gz"
    bw_name = "/path/atac_peaks.bw"
    bw_methyl_name = "/path/methylation.bw"

    positives_bed = "/path/positives_2000.bed"
    negatives_bed = "/path/negatives_2000.bed"

    checkpoint_path = "/checkpoints/pretrained.ckpt"

    tokenizer = DNATokenizerHF()

    positives = load_intervals_per_split(positives_bed, ["chr19_hap1"])
    negatives = load_intervals_per_split(negatives_bed, ["chr19_hap1"])

    print(len(positives))
    print(len(negatives))

    random.seed(42)

    pos_subset = random.sample(positives, num_examples)
    neg_subset = random.sample(negatives, num_examples)
    
    test_dataset = ChromDataset(fasta_name, bw_name, bw_methyl_name, tokenizer, positives_intervals=pos_subset, negatives_intervals=neg_subset, outputlen=2000)
    test_loader = DataLoader(test_dataset, batch_size=100, collate_fn=partial(pad_collate_fn, pad_index=tokenizer.pad_token_id))
    
    config_args = get_config_base()
    model = BaseLightningModule.load_from_checkpoint(checkpoint_path, config=config_args, model_name="standard")
    model = model.to(device)

    embeddings_np, labels_np = extract_embeddings(model, test_loader, device)
    visualize_umap_result(embeddings_np, labels_np, "token")

    #embeddings_np_seq, labels_np_seq = extract_sequence_embeddings(model, test_loader, device)
    #print(embeddings_np_seq.shape)
    #visualize_umap_result(embeddings_np_seq, labels_np_seq, "sequence")

    print("Finish")
    

if __name__ == "__main__":
    main()
