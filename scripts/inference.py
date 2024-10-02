
"""
Inference directly from the pre-trained models. Incudes in-context learning.
Allows option to use CoT.
"""

import os
import sys
import argparse
from datetime import datetime
from time import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

from model_config import MODELS, bytes_to_gb
from templates import BASE_ICL, BASE_ANSWER_ICL, BASE_BOXED_ICL, COT_ICL, COT_ANSWER_ICL, COT_BOXED_ICL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def determine_in_context_learning(append_to_prompt, model_name):
    """
    Determine which in-context learning to use based on the model name.
    """
    if append_to_prompt.lower() == 'cot':
        if model_name in ['deepseekmath-7b']:
            return COT_BOXED_ICL
        elif model_name in ['metamathmistral-7b', 'wizardmath-7b']:
            return COT_ANSWER_ICL
        else:
            return COT_ICL
    elif append_to_prompt.lower() == 'base':
        if model_name in ['deepseekmath-7b']:
            return BASE_BOXED_ICL
        elif model_name in ['metamathmistral-7b', 'wizardmath-7b']:
            return BASE_ANSWER_ICL
        else:
            return BASE_ICL
    elif append_to_prompt.lower() == 'none':
        return []
    else:
        raise ValueError("Invalid value for append_to_prompt. Choose from 'cot', 'base', 'none'")

def process_datafile(file, tokenizer, icl_prompt):
    """
    Process the datafile to create prompts for the model.    
    """
    df = pd.read_csv(file)
    df = df[df.Split == 'test']

    def create_prompt(row):
        res = icl_prompt.copy()
        res.append({"role": "user", "content": f"[QUESTION] {row['Parsed_Problem']}"})
        return tokenizer.apply_chat_template(res, add_generation_prompt=True, tokenize=False)
    
    df['prompt'] = df.apply(create_prompt, axis=1)
    return df

def find_offset(icl_prompt, response):
    """
    Find the offset to remove the in-context learning part of the response.
    """
    return response.find(icl_prompt[-1]['content']) + len(icl_prompt[-1]['content'])
 
def generate_responses(model, tokenizer, df, batch_size, max_length, sample):
    """
    Generate responses from the model. Returns tokenized responses.
    """
    def preprocess(examples):
        return tokenizer(examples['prompt'], truncation=True, add_special_tokens=False)

    print("## PROCESSING DATA ##")
    len_df = len(df)
    dataset = Dataset.from_pandas(df, split="test")
    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    model.config.use_cache = True
    model.eval()
    
    print("## GENERATING RESPONSES ##")
    num_responses = 0
    responses = []
    full_start_time = time()
    for batch_num, batch in enumerate(dataloader):
        batch_start_time = time()
        input_ids = batch["input_ids"].to(device)
        num_responses += input_ids.shape[0]
        len_input_ids = input_ids.shape[1]
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=len_input_ids+max_length, pad_token_id=tokenizer.pad_token_id, do_sample=sample)

        responses.extend(outputs)
        if batch_num % 10 == 0:
            print(f"{num_responses} / {len_df} completed. Time taken for batch: {time() - batch_start_time:.2f}s")       
            sys.stdout.flush()

    print(f"Total time taken: {(time() - full_start_time)/60:.2f} minutes")
    print("## GENERATION COMPLETE ##", end = '\n\n')
    return responses

def main(args):
    """
    Main function to run inference.
    """
    to_sample = not args.greedy

    output_file = os.path.join(args.output, f"{args.model}_{args.run_num}.csv")
    
    if args.append:
        if not os.path.exists(output_file):
            raise ValueError("Output file does not exist")
        current_df = pd.read_csv(output_file)
    else: 
        if not os.path.exists(args.output):
            os.makedirs(args.output)
    
    model_config = MODELS[args.model]()
    print(f"Model size: {model_config.get_model_size()} GB")

    model = model_config.get_model()
    tokenizer = model_config.get_tokenizer()
    
    if args.append_to_prompt.lower() == 'none':
        if args.model_dir is None:
            raise ValueError("Model directory required for finetuned model")
        model_config.load_peft_model(args.model_dir)
        model = model_config.get_model()
        tokenizer = model_config.get_tokenizer()

    icl_prompt = determine_in_context_learning(args.append_to_prompt, args.model)
    df = process_datafile(args.datafile, tokenizer, icl_prompt)
    responses = generate_responses(model, tokenizer, df, args.batch_size, args.max_length, to_sample)
    df['Output'] = tokenizer.batch_decode(responses, skip_special_tokens=True)
    if args.append_to_prompt.lower() != 'none':
        df['Output'] = df['Output'].apply(lambda x: x[find_offset(icl_prompt, x)+1:])
    
    if args.append:
        current_df = current_df[df.columns] # Remove any extra columns
        df = pd.concat([current_df, df], ignore_index=True)

    df.to_csv(output_file, index=False)

    print(f"Output saved to {output_file}")
    print(f"Max GPU memory used: {bytes_to_gb(torch.cuda.max_memory_allocated())} GB")   

if __name__ == "__main__":
    from datetime import datetime

    now = datetime.now()    
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"Date/Time: {dt_string}", end ='\n\n')
    
    parser = argparse.ArgumentParser(description="Run inference on a model")
    parser.add_argument("--datafile", type=str, help="Path to the datafile", required=True)
    parser.add_argument("--output", type=str, help="Path to the output file", required=True)
    parser.add_argument("--model", type=str, help="Model to use", required=True)
    parser.add_argument("--model_dir", type=str, help="Path to the model directory", required=False)
    parser.add_argument("--batch_size", type=int, help="Batch size for inference", default=8)
    parser.add_argument("--max_length", type=int, help="Max length of the response", default=250)
    parser.add_argument("--append_to_prompt", type=str, help="What to append to prompt: 'cot', 'base', 'none'", default='none')
    parser.add_argument("--greedy", action='store_true', help="Disable sampling", default=False)
    parser.add_argument("--run_num", type=int, help="Run number", default=0)
    parser.add_argument("-a", "--append", help="Append to the output file", action="store_true", default=False)

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

        

        


