
import os
import argparse
from datetime import datetime
from time import time

import pandas as pd
import re
import sympy as sp


SUBSTITUTIONS = {
    '\\boxed': '',
    '\\cdot': '*',
    '\\displaystyle': '',
    '\\div': '/',
    '\\geq': '>=',
    '\\left': '',
    '\\leq': '<=',
    '\\right': '',
    '\\text': '',
    '\\times': '*',
    '\\$': '',
    '\\,': '',
    '\\(': '(',
    '\\)': ')',
    '\\[': '',
    '\\]': '',
    '(Choice A)': '',
    '(Choice B)': '',
    '(Choice C)': '',
    '(Choice D)': '',
    '(Choice E)': '',
    'Choice A:': '',
    'Choice B:': '',
    'Choice C:': '',
    'Choice D:': '',
    'Choice E:': '',
    'x 10^': '* 10^',
    'remainder': 'R',
    '$': '',
}   

LATER_SUBSTITUTIONS = {
    'Choice A)': '',
    'Choice B)': '',
    'Choice C)': '',
    'Choice D)': '',
    'Choice E)': '',
    '[': '',
    ']': '',
}

def substitute_answers(answer):
    """
    Perform substitutions on the answer text
    """
    for key, value in SUBSTITUTIONS.items():
        answer = answer.replace(key, value)

    for key, value in LATER_SUBSTITUTIONS.items():
        answer = answer.replace(key, value)
    
    # \dfrac{N}{M} -> N/M
    pattern = r'\\dfrac{([^{}]+)}{([^{}]+)}'
    answer = re.sub(pattern, r'((\1)/(\2))', answer)

    # \frac{N}{M} -> N/M
    pattern = r'\\frac{([^{}]+)}{([^{}]+)}'
    answer = re.sub(pattern, r'((\1)/(\2))', answer)

    # \overline{N} -> Recurring N
    pattern = r'\\overline{(\d+)}'
    answer = re.sub(pattern, r'\1, where \1 is recurring ', answer)

    answer = answer.replace('{', '')
    answer = answer.replace('}', '')

    return answer


def deepseekmath_postprocess(expression):
    try:
        expression = expression.split('The answer is')[1].strip().split('\n')[0].strip()
        if expression[-1] == '.': 
            expression = expression[:-1]
    except:
        expression = expression
    return expression

def mcq_postprocess(expression, question):
    if 'Choice A' in question:
        generated_answer = f"(Choice {expression})"
        try:
            expression = question.split(generated_answer)[1].split('(Choice')[0].strip()
        except:
            expression = expression
    return expression

def metamath_postprocess(expression):
    expression = expression.replace('industries', '')
    expression = expression.replace('Industries', '')
    try:
        expression = expression.split('The answer is')[1].strip().split('\n')[0].strip()
        expression = expression.split('###')[0].strip()
        if expression[-1] == '.': 
            expression = expression[:-1]
    except:
        expression = expression
    return expression

def mistral_postprocess(expression):
    return expression.replace('industries', '')

def phi_2_postprocess(expression):
    try:
        expression = expression.split('Output: \n')[1].split('\n')[0].strip()
    except:
        expression = expression
    return expression

def rho_1b_postprocess(expression):
    expression = expression.split('is the answer')[0].strip()
    try:
        expression = expression.split('the answer is')[1].strip().split('\n')[0].strip()
    except:
        expression = expression
    return expression

def wizardmath_postprocess(expression):
    try:
        expression = expression.split('The answer is')[1].strip().split('\n')[0].strip()
        expression = expression.split('###')[0].strip()
        if expression[-1] == '.': 
            expression = expression[:-1]
    except:
        expression = expression
    return expression

def qwen_7b_postprocess(expression):
    if 'simplified to' in expression:
        expression = expression.split('simplified to')[1].strip()
    elif 'simplifies to' in expression:
        expression = expression.split('simplifies to')[1].strip()
    elif 'simplified to' in expression:
        expression = expression.split('simplified to')[1].strip()
    elif 'simplified further to' in expression:
        expression = expression.split('simplified further to')[1].strip()
    elif 'or equivalently' in expression:
        expression = expression.split('or equivalently')[1].strip()
    return expression


def retrieve_answer(row, model_name):
    expression = row.Output
    expression = expression.replace('\\r', '')

    if model_name == 'deepseekmath-7b':
        expression = deepseekmath_postprocess(expression)
    elif model_name in ['gpt3', 'gpt4o', 'gpt4omini']:
        expression = expression.replace('\n', '')
    elif model_name == 'metamathmistral-7b':
        expression = metamath_postprocess(expression)
    elif model_name == 'mistral-7b':
        expression = mistral_postprocess(expression)
    elif model_name == 'phi-2':
        if '[ANSWER]' not in expression:
            expression = phi_2_postprocess(expression)
    elif model_name == 'wizardmath-7b':
        if '[ANSWER]' not in expression:
            expression = wizardmath_postprocess(expression)
    
    try:
        expression = expression.split('[ANSWER]')[1].split('[')[0].split(']')[0].split('||')[0].split('\n')[0].strip()
    except:
        expression = expression

    if model_name == 'qwen-7b':
        expression = qwen_7b_postprocess(expression)

    if model_name == 'rho-1b':
        expression = rho_1b_postprocess(expression)

    if not row.Equation_Answer:
       expression = expression.split('=')[-1]

    # Substitute the answer
    expression = substitute_answers(expression)

    expression = expression.split('\r')[0].split('\n')[0].strip()
    expression = expression.replace(':', '')
    expression = expression.replace(',', '')
    expression = expression.replace('\\n', '')
    expression = expression.replace('\\', '')
    if row.Integer_Answer:
        pattern = r'[^-0-9.]'
        expression = re.sub(pattern, '', expression)
    
    expression = expression.strip()
    expression = mcq_postprocess(expression, row.Parsed_Problem)

    return expression

def preprocess_expression(expression):
    if expression[-1] == '.': 
        expression = expression[:-1]
    expression = re.sub(r'(\d)([a-df-zA-DF-Z])', r'\1*\2', expression)   # 3y -> 3*y
    expression = re.sub(r'(\d)\(([\s+\-*0-9a-zA-Z]+)\)', r'\1*(\2)', expression)   # 3(y) -> 3*y
    expression = re.sub(r'\(\(\s*(\d+)\s*\)/\s*\((\d+)\s*\)\)([a-zA-Z])', r'(\1/\2)*\3', expression)   # ((3)/(5))y -> (3/5)*y
    expression = re.sub(r'(\d+)\s*\(\(\s*(\d+)\s*\)/\s*\((\d+)\s*\)\)', r'(\1 + (\2/\3))', expression) # 2((1/2)) -> (2 + (1/2))
    expression = re.sub(r'(\d+)\s*\(([\da-zA-Z\+\-*/]+)\)', r'(\1*(\2))', expression) # 2((1/2)) -> (2 + (1/2))
    return expression
    
def calculator(expression):
    try:
        preprocessed_expression = preprocess_expression(expression)
        sympy_expr = sp.sympify(preprocessed_expression)
        simplified_expr = sp.simplify(sympy_expr)
        return simplified_expr
    except:
        print("Invalid expression: ", expression)
        return expression.replace(' ', '')
    
def check_equality(row):
    ans1 = row.Answer_Evaluated
    ans2 = row.Output_Evaluated
    if type(ans1) == str or type(ans2) == str:
        return ans1 == ans2
    try:
        return bool(sp.Equality(ans1, ans2))
    except:
        print(row.Parsed_Problem)
        print("ERROR IN CHECKING EQUALITY:", ans1, ans2)
    return False
     
def main(args):
    df = pd.read_csv(args.file)
    df['Equation_Answer'] = df.Parsed_Answer.str.contains('=')
    df['Generated_Answer'] = df.apply(retrieve_answer, axis=1, model_name=args.model)
    df['Answer_Evaluated'] = df['Parsed_Answer'].apply(calculator)
    df['Output_Evaluated'] = df['Generated_Answer'].apply(calculator)
    df['Correct'] = df.apply(check_equality, axis=1)

    out_df = pd.read_csv(args.output)
    if args.run_num == -1:
        new_column_prefix = f"{args.technique}_{args.model}"
    else:
        new_column_prefix = f"{args.technique}_{args.run_num}_{args.model}"
    out_df['Equation_Answer'] = df['Equation_Answer']
    out_df['Answer_Evaluated'] = df['Answer_Evaluated']
    out_df[f"{new_column_prefix}_Output"] = df['Output_Evaluated']
    out_df[f"{new_column_prefix}_Correct"] = df['Correct']
    df.to_csv(args.file, index=False)
    out_df.to_csv(args.output, index=False)

    print(f"\nResults saved to {args.file} and {args.output}")

if __name__ == "__main__":
    start = datetime.now()    
    dt_string = start.strftime("%d/%m/%Y %H:%M:%S")
    print(f"Date/Time: {dt_string}", end ='\n\n')

    parser = argparse.ArgumentParser(description="Evaluate the answers extracted by the model")
    parser.add_argument("--results_folder", type=str, help="The path to the file containing the text", required=True)
    parser.add_argument("--technique", type=str, required=True)
    parser.add_argument("--model", type=str, help="The model name", required=True)
    parser.add_argument("--output", type=str, help="The path to the consolidated output file", required=True)
    parser.add_argument("--run_num", type=int, help="The run number", required=False, default=0)
    args = parser.parse_args()

    if args.run_num == -1:
        args.file = os.path.join(args.results_folder, args.technique, f"{args.model}.csv")
    else:
        args.file = os.path.join(args.results_folder, args.technique, f"{args.model}_{args.run_num}.csv")
    main(args)
    
    end = datetime.now()
    print("\n\nTime taken:", end - start)
    print(f"Date/Time: {dt_string}")