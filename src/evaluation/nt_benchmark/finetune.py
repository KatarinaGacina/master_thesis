import torch
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from datasets import Dataset
from evaluation.nt_benchmark.data import get_train_dataset, get_test_dataset
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import copy
import argparse
import math
import random
from data.tokenizer.tokenizer import DNATokenizerHF
from evaluation.nt_benchmark.finetune_config import get_config_finetune, get_config_task
from model.model import FinetuneModel, FinetuneLongModel

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def next_multiple_of_128(L):
    return math.ceil(L / 128) * 128

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)

    args = parser.parse_args()

    print("Task:", args.task)
    task = args.task

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config_params = get_config_finetune()
    config_task = get_config_task(task)
    config_params.update(config_task)

    task = config_params["task"]

    train_dataset_all = get_train_dataset(task)
    test_dataset = get_test_dataset(task)

    split_dataset = train_dataset_all.train_test_split(
        test_size=0.05,
        seed=seed
    )

    train_dataset = split_dataset["train"]
    train_lengths = [len(x["sequence"]) for x in train_dataset]
    print(f"Unique lengths: {set(train_lengths)}")
    validation_dataset = split_dataset["test"]
    val_lengths = [len(x["sequence"]) for x in validation_dataset]
    print(f"Unique lengths: {set(val_lengths)}")

    #print(train_dataset[0])
    #print(validation_dataset[0])
    
    print("Data loaded.")

    tokenizer = DNATokenizerHF()
    config_params["vocab_size"] = tokenizer.vocab_size
    config_params["pad_index"] = tokenizer.pad_token_id

    def tokenize_function(examples):
        if config_params["model_type"] == "longcontext":
            return tokenizer(examples["sequence"], add_special_tokens=False, padding='max_length', max_length=next_multiple_of_128(config_params["outputlen"]), padding_side="right")
        else:
            return tokenizer(examples["sequence"], add_special_tokens=False, padding='max_length', max_length=config_params["outputlen"], padding_side="right")

    tokenized_train = train_dataset.map(tokenize_function)
    tokenized_val = validation_dataset.map(tokenize_function)
    tokenized_test = test_dataset.map(tokenize_function)

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    batch_size = config_params["batch_size"]
    eval_batch_size = config_params["eval_batch_size"]

    train_dataloader = DataLoader(
        tokenized_train, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_dataloader = DataLoader(
        tokenized_val, 
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
    )
    test_dataloader = DataLoader(
        tokenized_test, 
        batch_size=eval_batch_size, 
        shuffle=False,
        num_workers=2,
    )

    if config_params["model_type"] == "longcontext":
        print("Longcontext")
        model = FinetuneLongModel(config_params, config_params["pretrained"])
    else:
        model = FinetuneModel(config_params, config_params["pretrained"])
    model.to(device)


    num_epochs = config_params["num_epochs"]
    learning_rate = config_params["learning_rate"]
    logging_steps = config_params["logging_steps"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_score = -float('inf')
    patience = 0
    global_step = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :].bool().to(device)
            
            logits = model(inputs, attn_mask=attention_mask)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1

            if global_step % logging_steps == 0:
                model.eval()

                all_preds, all_labels = [], []
                total_val_loss = 0.0
                num_val_batches = 0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_inputs = val_batch['input_ids'].to(device)
                        val_labels = val_batch['labels'].to(device)
                        val_mask = val_batch.get('attention_mask', None)
                        if val_mask is not None:
                            val_mask = val_mask[:, None, None, :].bool().to(device)

                        val_logits = model(val_inputs, attn_mask=val_mask)
                        val_loss = criterion(val_logits, val_labels)
                        
                        total_val_loss += val_loss.item()
                        num_val_batches += 1

                        preds = torch.argmax(val_logits, dim=-1)
                        all_preds.append(preds.cpu())
                        all_labels.append(val_labels.cpu())

                avg_val_loss = total_val_loss / num_val_batches

                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)

                metric_name = config_params.get("metric", "accuracy")
                if metric_name == "f1":
                    score_value = f1_score(all_labels.numpy(), all_preds.numpy(), average='binary')
                elif metric_name == "mcc":
                    score_value = matthews_corrcoef(all_labels.numpy(), all_preds.numpy())
                else:
                    score_value = accuracy_score(all_labels.numpy(), all_preds.numpy())

                print(f"Step {global_step}, Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}, {metric_name.upper()}: {score_value:.4f}")

                if avg_val_loss < best_val_loss - 1e-4:
                    patience = 0
                    best_val_loss = avg_val_loss
                else:
                    patience += 1

                if score_value > best_score:
                    best_score = score_value
                    best_model_weights = copy.deepcopy(model.state_dict())
                    print(f"New best {metric_name.upper()}! Model saved at step {global_step}")

                model.train()

                if patience >= 10:
                    print(f"Early stopping triggered at step {global_step}")
                    break

        if patience >= 10:
            break
            

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    model_type = config_params["model_type"]
    output_path = config_params["output_path"]
    torch.save(model.state_dict(), f"{output_path}/{model_type}-{task}-best-model.pth")

    #evaluation
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :].bool().to(device)

            logits = model(inputs, attn_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    if config_params["metric"] == "f1":
        test_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='binary')
        print(f"Test F1: {test_f1:.4f}")
    elif config_params["metric"] == "mcc":
        test_mcc = matthews_corrcoef(all_labels.numpy(), all_preds.numpy())
        print(f"Test MCC: {test_mcc:.4f}")
    else:
        test_accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
        print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
    