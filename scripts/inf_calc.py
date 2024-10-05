
"""
Inference which makes use of tags to perform calculations.
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
import sympy as sp
import re

from model_config import MODELS, bytes_to_gb
from templates import TAGS_ICL


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_datafile(file, tokenizer, icl):
    """
    Process the datafile to create prompts for the model.    
    """
    df = pd.read_csv(file)
    df = df[df.Split == 'test']

    def create_prompt(row):
        res = []
        if icl:
            res = TAGS_ICL.copy()
        res.append({"role": "user", "content": f"[QUESTION] {row['Problem']}"})
        return tokenizer.apply_chat_template(res, add_generation_prompt=True, tokenize=False)
    
    df['prompt'] = df.apply(create_prompt, axis=1)
    return df

def find_offset(response):
    """
    Find the offset to remove the in-context learning part of the response.
    """
    return response.find(TAGS_ICL[-1]['content']) + len(TAGS_ICL[-1]['content'])
    
def preprocess_expression(expression):
    expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)   # 3y -> 3*y
    expression = re.sub(r'(\d)\(([a-zA-Z])\)', r'\1*\2', expression)   # 3(y) -> 3*y
    expression = re.sub(r'\(\(\s*(\d+)\s*\)/\s*\((\d+)\s*\)\)([a-zA-Z])', r'(\1/\2)*\3', expression)   # ((3)/(5))y -> (3/5)*y
    expression = re.sub(r'(\d+)\s*\(\(\s*(\d+)\s*\)/\s*\((\d+)\s*\)\)', r'(\1 + (\2/\3))', expression) # 2((1/2)) -> (2 + (1/2))
    expression = re.sub(r'(\d+)\s*\(([\da-zA-Z\+\-*/]+)\)', r'(\1*(\2))', expression) # 2((1/2)) -> (2 + (1/2))
    return expression

def calculator(expression):
    preprocessed_expression = preprocess_expression(expression)
    sympy_expr = sp.sympify(preprocessed_expression)
    simplified_expr = sp.simplify(sympy_expr)
    return str(simplified_expr)

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

    STARTWORK_TOKENS = tokenizer.convert_tokens_to_ids('||STARTWORK||')
    ENDWORK_TOKENS = tokenizer.convert_tokens_to_ids('||ENDWORK||')


    def generate_until_stop_token(input_ids, attention_mask, max_length, end_token_id):
        with torch.no_grad():
            return model.generate(input_ids, 
                                  attention_mask=attention_mask,
                                  max_length=max_length, 
                                  do_sample=sample, 
                                  pad_token_id=tokenizer.pad_token_id, 
                                  eos_token_id=[end_token_id, tokenizer.eos_token_id])

    def process_calculations(current_ids):
        start_indices = (current_ids == STARTWORK_TOKENS).nonzero(as_tuple=True)[0]
        end_indices = (current_ids == ENDWORK_TOKENS).nonzero(as_tuple=True)[0]

        if len(start_indices) > 0 and len(end_indices) > 0:
            last_start_idx = start_indices[-1].item()
            last_end_idx = end_indices[-1].item()

            extracted_content_ids = current_ids[last_start_idx+1:last_end_idx]
            extracted_content = tokenizer.decode(extracted_content_ids, skip_special_tokens=True)

            try:
                postprocessed_content = " "
                postprocessed_content += calculator(extracted_content)
                postprocessed_content += " ||ENDCALC|| "
                postprocessed_id = tokenizer.encode(postprocessed_content, add_special_tokens=False, return_tensors='pt').to(device)
                current_ids = torch.cat([current_ids, postprocessed_id[0]], dim=-1)
            except Exception as e:
                print(f"Postprocessing error: {e} | Content: {extracted_content}")
                current_ids = current_ids

        return current_ids

    def generate_one_batch(input_ids, attention_mask, max_length):
        n_samples = len(input_ids)
        original_input_ids = input_ids
        outputs = dict()
        indices = [i for i in range(len(input_ids))]
    
        while len(input_ids):
            try:
                input_ids = generate_until_stop_token(input_ids, attention_mask, max_length, ENDWORK_TOKENS)
        
                regenerate_ids = []
                regenerate_idx = []
                for i in range(len(input_ids)):
                    current_input_ids = input_ids[i]
                    reversed_ids = torch.flip(current_input_ids, dims=[0])
                    first_non_pad_index = (reversed_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item()
                    first_non_pad_value = current_input_ids[-(first_non_pad_index + 1)].item()
        
                    if first_non_pad_value == ENDWORK_TOKENS :
                        if first_non_pad_index == 0:
                            new_input_ids = process_calculations(current_input_ids)
                        else:
                            new_input_ids = process_calculations(current_input_ids[:-(first_non_pad_index)])
                        regenerate_ids.append(new_input_ids)
                        regenerate_idx.append(indices[i])
                    else:
                        outputs[indices[i]] = current_input_ids

                if regenerate_ids:
                    def remove_leading_pad(tensor):
                        non_pad_idx = 0
                        while non_pad_idx < len(tensor) and tensor[non_pad_idx] == tokenizer.pad_token_id:
                            non_pad_idx += 1
                        return tensor[non_pad_idx:]
        
                    processed_ids = [remove_leading_pad(tensor) for tensor in regenerate_ids]
                    longest = min(max(len(tensor) for tensor in processed_ids), max_length)
                    
                    padded_tensors = []
                    attention_masks = []
                    for i, tensor in enumerate(processed_ids):
                        pad_length = longest - len(tensor)
                        if pad_length < 0 or len(tensor) == max_length:
                            outputs[indices[i]] = current_input_ids
                            regenerate_idx.remove(indices[i])
                            continue
                        elif pad_length > 0:
                            padded_tensor = torch.cat([torch.tensor([tokenizer.pad_token_id]*pad_length).to(device), tensor], dim=0)
                            attn = torch.cat([torch.tensor([0] * pad_length).to(device), torch.tensor([1] * len(tensor)).to(device)], dim=0)
                        else:
                            padded_tensor = tensor
                            attn = torch.tensor([1] * len(tensor)).to(device)

                        padded_tensors.append(padded_tensor)
                        attention_masks.append(attn)
                    if padded_tensors:
                        input_ids = torch.stack(padded_tensors)
                        attention_mask = torch.stack(attention_masks)
                        indices = regenerate_idx
                    else:
                        break
                else:
                    break
            except Exception as e:
                print(f"Error in generating outputs. {e}")
                break
    
        final_output = []
        for i in range(n_samples):
            if i in outputs:
                final_output.append(outputs[i])
            else:
                final_output.append(original_input_ids[i])

        longest_length = max(tensor.size(0) for tensor in final_output)
        def pad_tensor(tensor, max_length, pad_token_id):
            pad_size = max_length - tensor.size(0)
            padding = torch.full((pad_size,), pad_token_id).to(device)
            return torch.cat((padding, tensor), dim=0)
        padded_output = [pad_tensor(tensor, max_length, tokenizer.pad_token_id) for tensor in final_output]
        final_output = torch.stack(padded_output)
        return final_output        
     

    model.config.use_cache=True
    model.eval() 

    print("## GENERATING RESPONSES ##")
    num_responses = 0
    responses = []
    full_start_time = time()
    for batch_num, batch in enumerate(dataloader):
        batch_start_time = time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch['attention_mask'].to(device)
        num_responses += input_ids.shape[0]
        len_input_ids = input_ids.shape[1]
        outputs = generate_one_batch(input_ids, attention_mask, len_input_ids+max_length)

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
    
    if not args.use_icl:
        if args.model_dir is None:
            raise ValueError("Model directory required for finetuned model")
        model_config.load_peft_model(args.model_dir)
        model = model_config.get_model()    
        tokenizer = model_config.get_tokenizer()

    df = process_datafile(args.datafile, tokenizer, args.use_icl)
    responses = generate_responses(model, tokenizer, df, args.batch_size, args.max_length, to_sample)
    df['Output'] = tokenizer.batch_decode(responses, skip_special_tokens=True)
    if args.use_icl:
        df['Output'] = df['Output'].apply(lambda x: x[find_offset(x)+1:])
    
    if args.append:
        current_df = current_df[df.columns] 
        df = pd.concat([current_df, df], ignore_index=True)
        
    df.to_csv(output_file, index=False)

    print(f"Output saved to {output_file}")
    print(f"Max GPU memory used: {bytes_to_gb(torch.cuda.max_memory_allocated())}")

if __name__ == "__main__":
    now = datetime.now()    
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"Date/Time: {dt_string}", end ='\n\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, required=True, help="Path to the datafile")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--model_dir", type=str, help="Path to the model directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max length of the response")
    parser.add_argument("--use_icl", action='store_true', help="Use in-context learning", default=False)
    parser.add_argument("--run_num", type=int, default=0, help="Run number")
    parser.add_argument("--greedy", action='store_true', help="Disable sampling", default=False)
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