
"""
Fine-tuning a pre-trained language model based on the explanations provided by gpt-3.5-turbo.

This handles the following setups:
- Direct fine-tuning of the model with Question, Explanation, and Answer.
- Fine-tuning with the explanations using the calculator
- Fine-tuning with including the topic in the explanation
"""


import os
import sys
import argparse
import json
from datetime import datetime
from time import time

import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb

from model_config import MODELS, bytes_to_gb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_datafile(file, use_topic):
    """
    Process the datafile to create prompts for the model.    
    """
    df = pd.read_csv(file)
    df = df[df.Split == 'train']

    def create_prompt(row):
        if use_topic:
            return [
                {'role': 'user', 'content': f"[QUESTION] {row['Problem']}"},
                {'role': 'assistant', 'content': f"[EXPLANATION] The topic is: {row['Exercise_Name']}. {row['Solution']}\n[ANSWER] {row['Answer']}\n"},
            ]
        else:
            return [
                {'role': 'user', 'content': f"[QUESTION] {row['Problem']}"},
                {'role': 'assistant', 'content': f"[EXPLANATION] {row['Solution']}\n[ANSWER] {row['Answer']}\n"},
            ]  
    
    df['prompt'] = df.apply(create_prompt, axis=1)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Exercise_Name'], random_state=42)
    return train_df, val_df
    
def load_config(config_file):
    """
    Load the configuration file.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def create_model(config, use_calculator):
    """
    Create the LoRA model based on the configuration.
    Returns the model and the tokenizer.
    """
    model_config = MODELS[config.model]()
    if use_calculator:
        print("Adding calculator tokens...")
        model_config.add_calculator_tokens()
    print(f"Model size: {model_config.get_model_size()} GB")
    model = model_config.get_model()
    tokenizer = model_config.get_tokenizer()

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if hasattr(config, 'lora_target_modules'):
        lora_config = LoraConfig(
            r = config.lora_r,
            lora_alpha=config.lora_alpha,
            bias='none',
            lora_dropout=config.lora_dropout,
            task_type='CAUSAL_LM',
            target_modules=config.lora_target_modules
        )
    else:
        lora_config = LoraConfig(
            r = config.lora_r,
            lora_alpha=config.lora_alpha,
            bias='none',
            lora_dropout=config.lora_dropout,
            task_type='CAUSAL_LM',
        )

    model = get_peft_model(model, lora_config)
    model.to(device)
    return model, tokenizer

def load_model(config, model_dir): 
    model_config = MODELS[config.model]()
    model_config.load_peft_model(model_dir)
    model = model_config.get_model()
    tokenizer = model_config.get_tokenizer()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print(f"Model size: {model_config.get_model_size()} GB")
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
    model.config.use_cache = False
    model.to(device)
    return model, tokenizer

def generate_datasets(train_df, val_df, tokenizer):
    """
    Generate the datasets using the train and validation dataframes.
    """
    train_dataset = Dataset.from_pandas(train_df, split="train", preserve_index=True)
    val_dataset = Dataset.from_pandas(val_df, split="validation", preserve_index=True)
    
    def tokenize_function(examples):
        return tokenizer(tokenizer.apply_chat_template(examples['prompt'], tokenize=False), truncation=True, add_special_tokens=True)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
    
    return train_dataset, val_dataset

def split_into_blocks(train_ds, val_ds, block_size):
    """
    Split the tokenized examples into blocks of fixed length.
    """
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    train_ds = train_ds.map(group_texts, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(group_texts, batched=True, remove_columns=val_ds.column_names)

    train_ds.set_format(type='torch')
    val_ds.set_format(type='torch')

    return train_ds, val_ds

def create_dataloader(train_ds, val_ds, tokenizer, batch_size):
    """
    Create the DataLoader for the train and validation datasets.
    """
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=data_collator, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=data_collator, shuffle=False)
    return train_loader, val_loader

def finetune_model(model, train_loader, val_loader, config, save_dir):
    """
    Fine-tune the model using the train and validation datasets.
    """
    wandb.init(**config)
    config = wandb.config

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    print("\n## START FINETUNING ##")
    nb_seen = 0
    best_val_loss = float('inf')
    early_stopping = 0
    overall_start_time = time()
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        training_start_time = time()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            if step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            nb_seen += input_ids.shape[0]

            if step % config.logging_steps == 0:
                metrics = {'train/train_loss': total_loss / (step + 1), 'train/epoch': epoch + 1, 'train/step': step + 1, 'nb_seen': nb_seen}
                wandb.log(metrics, commit=True)
                print(f"Epoch {epoch + 1} - Progress: {step}/{len(train_loader)} - Loss: {metrics['train/train_loss']} - Time: {time() - training_start_time:.2f} seconds")
                sys.stdout.flush()

        training_time = time() - training_start_time
        metrics = {'train/train_loss': total_loss / len(train_loader), 'train/epoch': epoch + 1, 'train/total_time': training_time, 'nb_seen': nb_seen}
        wandb.log(metrics, commit=True)
        print(f"Epoch {epoch + 1} - Training Loss: {metrics['train/train_loss']} - Time: {training_time/60:.2f} minutes")

        model.eval()
        total_loss = 0
        val_start_time = time()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

        val_time = time() - val_start_time
        val_loss = total_loss / len(val_loader)
        val_metrics = {'eval/val_loss': val_loss, 'eval/epoch': epoch + 1, 'eval/total_time': val_time}
        print(f"Epoch {epoch + 1} - Validation Loss: {val_metrics['eval/val_loss']} - Time: {val_time / 60:2f} minutes")
        sys.stdout.flush()
        wandb.log({**metrics, **val_metrics}, commit=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(save_dir)
            print(f"Model saved to {save_dir}")
        else:
            early_stopping += 1
            if early_stopping == 2: 
                print(f"Early stopping. Best validation loss: {best_val_loss}")
                break

        scheduler.step()

    overall_time = time() - overall_start_time
    print(f"Total time taken: {overall_time / 60:.2f} minutes")
    print("## FINETUNING COMPLETE ##", end='\n\n')
    sys.stdout.flush()
    wandb.finish()  
    return model
            
def main(args):
    """
    Main function to run the fine-tuning.
    """
    config = load_config(args.config)

    class jsonConfig:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    json_config = jsonConfig(config['config'])

    TIMESTAMP = datetime.now().strftime("%y%m%d-%H%M")
    config['name'] = f"{json_config.model}_{TIMESTAMP}"
    print("## CONFIGURATIONS ##")
    print(config, end = '\n\n')

    train_df, val_df = process_datafile(args.datafile, args.use_topic)
    if args.model_dir is not None:
        model, tokenizer = load_model(json_config, args.model_dir)
    else:
        model, tokenizer = create_model(json_config, args.use_calculator)
    train_ds, val_ds = generate_datasets(train_df, val_df, tokenizer)
    train_ds, val_ds = split_into_blocks(train_ds, val_ds, json_config.block_size)
    train_loader, val_loader = create_dataloader(train_ds, val_ds, tokenizer, json_config.batch_size)
    MODEL_SAVE_DIR = os.path.join(args.output, config['name'])
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    model = finetune_model(model, train_loader, val_loader, config, MODEL_SAVE_DIR)

    print(f"Model saved to {MODEL_SAVE_DIR}")
    print(f"Max GPU memory used: {bytes_to_gb(torch.cuda.max_memory_allocated())} GB")
    sys.stdout.flush()

if __name__ == "__main__":
    now = datetime.now()    
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"Date/Time: {dt_string}", end ='\n\n')

    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained language model.")
    parser.add_argument("--datafile", type=str, help="Path to the datafile.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    parser.add_argument("--output", type=str, help="Output directory.")
    parser.add_argument("--use_topic", action='store_true', help="Include the topic in the explanation.", default=False)
    parser.add_argument("--use_calculator", action='store_true', help="Use the calculator for fine-tuning.", default=False)
    parser.add_argument("--model_dir", type=str, help="Path to the model directory.", default=None, required=False)

    print(f"Device: {device}")
    try:
        print(torch.cuda.get_device_properties(device))   
    except:
        print("No GPU found")
    print()

    args = parser.parse_args()
    print("## ARGUMENTS PROVIDED ##")
    print(args)
    print()
    sys.stdout.flush()
    main(args)

    now = datetime.now()    
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"Date/Time: {dt_string}")
    sys.stdout.flush()
