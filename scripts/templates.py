
"""
Contains chat templates for each model.
Contains examples for in-context learning.
"""

## Chat Templates

DEEPSEEK_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'].strip()}}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant: ' + message['content'].strip() + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' Please reason step by step, and put your final answer within \\\\boxed{}\n\nAssistant:' }}{% endif %}"""

FALCON_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '>>QUESTION<<\\n' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ '>>ANSWER<<\\n' + message['content'].strip() + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '>>ANSWER<< ' }}{% endif %}{% endfor %}"""

GEMMA_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"""

METAMATH_TEMPLATE = """{{ 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' }}{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '### Instruction:\n' + message['content'].strip() + '\n'}}{% elif message['role'] == 'assistant' %}{{ '### Response: ' + message['content'].strip() + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '### Response: Let\\'s think step by step.' }}{% endif %}{% endfor %}"""

MISTRAL_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"""

PHI2_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + 'Instruct:\\n' + message['content'].strip() }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Output:\\n'  + message['content'].strip() + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ 'Output: ' }}{% endif %}{% endfor %}"""

TINYLLAMA_TEMPLATE = """{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"""

VICUNA_TEMPLATE= """{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'].strip() + ' '}}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'].strip() + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% if loop.last and add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}{% endfor %}"""

## In-Context Learning Examples
BASE_ICL = [
    {"role": "user", "content": "You are a helpful AI assistant, tasked with answering math questions. Provide your answer after [ANSWER].\n[QUESTION] ((5)/(9)) / ((8)/(5)) = ?"},
    {"role": "assistant", "content": "[ANSWER] ((25)/(72))"},
    {"role": "user", "content": "[QUESTION] There are 3 toy cars, 5 dolls, and 2 teddy bears on display. How many toys are there in total?"},
    {"role": "assistant", "content": "[ANSWER] 10"},
    {"role": "user", "content": "[QUESTION] Apply the distributive property to create an equivalent expression. 5(3p - q - 2r) = ?"},
    {"role": "assistant", "content": "[ANSWER] 15p - 5q - 10r"},
    {"role": "user", "content": "[QUESTION] Multiply. 324 * 21"},
    {"role": "assistant", "content": "[ANSWER] 6804"},
    {"role": "user", "content": "[QUESTION] Solve the equation. 6 = (x/4) x = ?"},
    {"role": "assistant", "content": "[ANSWER] 24"},
    {"role": "user", "content": "[QUESTION] Write your answer as a whole number and remainder. 61 / 7 = ?"},
    {"role": "assistant", "content": "[ANSWER] 8 R 5"},
    {"role": "user", "content": "[QUESTION] The bakery sold ((3)/(5))  of its cakes in the morning and ((1)/(4)) of the remaining cakes in the afternoon. How many cakes were sold altogether if there were initially 60 cakes?"},
    {"role": "assistant", "content": "[ANSWER] 42"},
    {"role": "user", "content": "[QUESTION] David rents a car for a day. The rental company charges an initial fee of 50 dollars plus 20 dollars for each hour the car is rented. If David pays 130 dollars for the rental, write a function C in dollars, where C(h) represents the total cost of renting the car based on the number of hours h it is rented."},
    {"role": "assistant", "content": "[ANSWER] 50 + 20h"}
]

BASE_BOXED_ICL = [
    {"role": "user", "content": "((5)/(9)) / ((8)/(5)) = ?"},
    {"role": "assistant", "content": "The answer is \\boxed{((25)/(72))}"},
    {"role": "user", "content": "There are 3 toy cars, 5 dolls, and 2 teddy bears on display. How many toys are there in total?"},
    {"role": "assistant", "content": "The answer is \\boxed{10}"},
    {"role": "user", "content": "Apply the distributive property to create an equivalent expression. 5(3p - q - 2r) = ?"},
    {"role": "assistant", "content": "The answer is \\boxed{15p - 5q - 10r}"},
    {"role": "user", "content": "Multiply. 324 * 21"},
    {"role": "assistant", "content": "The answer is \\boxed{6804}"},
    {"role": "user", "content": "Solve the equation. 6 = (x/4) x = ?"},
    {"role": "assistant", "content": "The answer is \\boxed{24}"},
    {"role": "user", "content": "Write your answer as a whole number and remainder. 61 / 7 = ?"},
    {"role": "assistant", "content": "The answer is \\boxed{8 R 5}"},
    {"role": "user", "content": "The bakery sold ((3)/(5))  of its cakes in the morning and ((1)/(4)) of the remaining cakes in the afternoon. How many cakes were sold altogether if there were initially 60 cakes?"},
    {"role": "assistant", "content": "The answer is \\boxed{42}"},
    {"role": "user", "content": "David rents a car for a day. The rental company charges an initial fee of 50 dollars plus 20 dollars for each hour the car is rented. If David pays 130 dollars for the rental, write a function C in dollars, where C(h) represents the total cost of renting the car based on the number of hours h it is rented."},
    {"role": "assistant", "content": "The answer is \\boxed{50 + 20h}"}
]

BASE_ANSWER_ICL =  [
    {"role": "user", "content": "((5)/(9)) / ((8)/(5)) = ?"},
    {"role": "assistant", "content": "The answer is ((25)/(72))"},
    {"role": "user", "content": "There are 3 toy cars, 5 dolls, and 2 teddy bears on display. How many toys are there in total?"},
    {"role": "assistant", "content": "The answer is 10"},
    {"role": "user", "content": "Apply the distributive property to create an equivalent expression. 5(3p - q - 2r) = ?"},
    {"role": "assistant", "content": "The answer is 15p - 5q - 10r"},
    {"role": "user", "content": "Multiply. 324 * 21"},
    {"role": "assistant", "content": "The answer is 6804"},
    {"role": "user", "content": "Solve the equation. 6 = (x/4) x = ?"},
    {"role": "assistant", "content": "The answer is 24"},
    {"role": "user", "content": "Write your answer as a whole number and remainder. 61 / 7 = ?"},
    {"role": "assistant", "content": "The answer is 8 R 5"},
    {"role": "user", "content": "The bakery sold ((3)/(5))  of its cakes in the morning and ((1)/(4)) of the remaining cakes in the afternoon. How many cakes were sold altogether if there were initially 60 cakes?"},
    {"role": "assistant", "content": "The answer is 42"},
    {"role": "user", "content": "David rents a car for a day. The rental company charges an initial fee of 50 dollars plus 20 dollars for each hour the car is rented. If David pays 130 dollars for the rental, write a function C in dollars, where C(h) represents the total cost of renting the car based on the number of hours h it is rented."},
    {"role": "assistant", "content": "The answer is 50 + 20h"}
]

COT_ICL = [
    {"role": "user", "content": "You are a helpful AI assistant, tasked with answering math questions. Provide a step-by-step explanation after [EXPLANATION], and your final answer after [ANSWER].\n[QUESTION] ((5)/(9)) / ((8)/(5)) = ?"},
    {"role": "assistant", "content": "[EXPLANATION] To divide fractions, we can multiply the first fraction by the reciprocal of the second fraction. So, (5/9) divided by (8/5) becomes (5/9) multiplied by (5/8), which simplifies to (25/72). [ANSWER] (25/72)"},

    {"role": "user", "content": "[QUESTION] There are 3 toy cars, 5 dolls, and 2 teddy bears on display. How many toys are there in total?"},
    {"role": "assistant", "content": "[EXPLANATION] To find the total number of toys, we add the quantities of each type of toy together: 3 toy cars + 5 dolls + 2 teddy bears = 10 toys in total. [ANSWER] 10"},

    {"role": "user", "content": "[QUESTION] Apply the distributive property to create an equivalent expression. 5(3p - q - 2r) = ?"},
    {"role": "assistant", "content": "[EXPLANATION] The distributive property distributes the factor outside the parentheses to each term inside. So, 5 times 3p is 15p, 5 times -q is -5q, and 5 times -2r is -10r, resulting in the equivalent expression 15p - 5q - 10r. [ANSWER] 15p - 5q - 10r"},

    {"role": "user", "content": "[QUESTION] Multiply. 324 * 21"},
    {"role": "assistant", "content": "[EXPLANATION] To multiply 324 by 21. Multiply 324 by 1 (the units digit of 21): 324 * 1 = 324. Multiply 324 by 20 (the tens digit of 21): 324 * 20 = 6480. Add the results from steps 1 and 2 to get the final product: 324 + 6480 = 6804. Therefore, 324 multiplied by 21 equals 6804. [ANSWER] 6804"},

    {"role": "user", "content": "[QUESTION] Solve the equation. 6 = (x/4) x = ?"},
    {"role": "assistant", "content": "[EXPLANATION] To solve for x, we multiply both sides of the equation by 4 to isolate x. This gives us x = 6 * 4, which simplifies to x = 24. [ANSWER] 24"},

    {"role": "user", "content": "[QUESTION] Write your answer as a whole number and remainder. 61 / 7 = ?"},
    {"role": "assistant", "content": "[EXPLANATION] When dividing 61 by 7, 7 goes into 61 8 times with a remainder of 5. [ANSWER] 8 R 5"},

    {"role": "user", "content": "[QUESTION] The bakery sold ((3)/(5))  of its cakes in the morning and ((1)/(4)) of the remaining cakes in the afternoon. How many cakes were sold altogether if there were initially 60 cakes?"},
    {"role": "assistant", "content": "[EXPLANATION] The bakery initially sold 3/5 * 60 = 36 cakes in the morning. Then, it sold 1/4 * (60 - 36) = 6 cakes in the afternoon. Adding these together gives us a total of 36 + 6 = 42 cakes sold. [ANSWER] 42"},

    {"role": "user", "content": "[QUESTION] David rents a car for a day. The rental company charges an initial fee of 50 dollars plus 20 dollars for each hour the car is rented. If David pays 130 dollars for the rental, write a function C in dollars, where C(h) represents the total cost of renting the car based on the number of hours h it is rented."},
    {"role": "assistant", "content": "[EXPLANATION] The total cost C consists of the initial fee of 50 dollars plus 20 dollars for each hour h the car is rented. Thus, the function C(h) = 50 + 20h. [ANSWER] 50 + 20h"}
]

TAGS_ICL = [
    {"role": "user", "content": "You are a helpful AI assistant, tasked with answering math questions. Provide a step-by-step explanation after [EXPLANATION], and your final answer after [ANSWER]. For any calculations, they should be provided between ||STARTWORK|| and ||ENDWORK|| tags, and the final answer between the ||ENDWORK|| and ||ENDCALC|| tag, where the answer should be a final simplified form of the calculation.\n[QUESTION] ((5)/(9)) / ((8)/(5)) = ?"},
    {"role": "assistant", "content": "[EXPLANATION] To divide fractions, we can multiply the first fraction by the reciprocal of the second fraction. So, (5/9) divided by (8/5) becomes ||STARTWORK|| (5/9) * (5/8) ||ENDWORK|| (25/72) ||ENDCALC||. [ANSWER] (25/72)"},

    {"role": "user", "content": "[QUESTION] There are 3 toy cars, 5 dolls, and 2 teddy bears on display. How many toys are there in total?"},
    {"role": "assistant", "content": "[EXPLANATION] To find the total number of toys, we add the quantities of each type of toy together: ||STARTWORK|| 3 + 5 + 2 ||ENDWORK|| 10 ||ENDCALC||. [ANSWER] 10"},

    {"role": "user", "content": "[QUESTION] Apply the distributive property to create an equivalent expression. 5(3p - q - 2r) = ?"},
    {"role": "assistant", "content": "[EXPLANATION] The distributive property distributes the factor outside the parentheses to each term inside. So, ||STARTWORK|| 5 * 3p ||ENDWORK|| 15p ||ENDCALC|| and ||STARTWORK|| 5 * -q ||ENDWORK|| -5q ||ENDCALC||, and ||STARTWORK|| 5 * -2r ||ENDWORK|| -10r ||ENDCALC||, resulting in the equivalent expression 15p - 5q - 10r. [ANSWER] 15p - 5q - 10r"},

    {"role": "user", "content": "[QUESTION] Multiply. 324 * 21"},
    {"role": "assistant", "content": "[EXPLANATION] To multiply 324 by 21. Multiply 324 by 1 (the units digit of 21): ||STARTWORK|| 324 * 1 ||ENDWORK|| 324 ||ENDCALC||. Multiply 324 by 20 (the tens digit of 21): ||STARTWORK|| 324 * 20 ||ENDWORK|| 6480 ||ENDCALC||. Add the results from steps 1 and 2 to get the final product: ||STARTWORK|| 324 + 6480 ||ENDWORK|| 6804 ||ENDCALC||. Therefore, 324 multiplied by 21 equals 6804. [ANSWER] 6804"},

    {"role": "user", "content": "[QUESTION] Solve the equation. 6 = (x/4) x = ?"},
    {"role": "assistant", "content": "[EXPLANATION] To solve for x, we multiply both sides of the equation by 4 to isolate x. This gives us x = 6 * 4. Solving for the right-hand side, we get ||STARTWORK|| 6 * 4 ||ENDWORK|| 24 ||ENDCALC||. Hence x = 24 [ANSWER] 24"},

    {"role": "user", "content": "[QUESTION] Write your answer as a whole number and remainder. 61 / 7 = ?"},
    {"role": "assistant", "content": "[EXPLANATION] When dividing 61 by 7, 7 goes into 61 8 times with a remainder of 5. [ANSWER] 8 R 5"},

    {"role": "user", "content": "[QUESTION] The bakery sold ((3)/(5))  of its cakes in the morning and ((1)/(4)) of the remaining cakes in the afternoon. How many cakes were sold altogether if there were initially 60 cakes?"},
    {"role": "assistant", "content": "[EXPLANATION] The bakery initially sold ||STARTWORK|| 3/5 * 60 ||ENDWORK|| 36 ||ENDCALC|| cakes in the morning. Then, it sold ||STARTWORK|| 1/4 * (60 - 36) ||ENDWORK|| 6 ||ENDCALC|| cakes in the afternoon. Adding these together gives us a total of ||STARTWORK|| 36 + 6 ||ENDWORK|| 42 ||ENDCALC|| cakes sold. [ANSWER] 42"},

    {"role": "user", "content": "[QUESTION] David rents a car for a day. The rental company charges an initial fee of 50 dollars plus 20 dollars for each hour the car is rented. If David pays 130 dollars for the rental, write a function C in dollars, where C(h) represents the total cost of renting the car based on the number of hours h it is rented."},
    {"role": "assistant", "content": "[EXPLANATION] The total cost C consists of the initial fee of 50 dollars plus 20 dollars for each hour h the car is rented. Thus, the function C(h) = 50 + 20h. [ANSWER] 50 + 20h"}
]

COT_BOXED_ICL = [
    {"role": "user", "content": "((5)/(9)) / ((8)/(5)) = ?"},
    {"role": "assistant", "content": "To divide fractions, we can multiply the first fraction by the reciprocal of the second fraction. So, (5/9) divided by (8/5) becomes (5/9) multiplied by (5/8), which simplifies to (25/72). The answer is \\boxed{25/72}"},

    {"role": "user", "content": "There are 3 toy cars, 5 dolls, and 2 teddy bears on display. How many toys are there in total?"},
    {"role": "assistant", "content": "To find the total number of toys, we add the quantities of each type of toy together: 3 toy cars + 5 dolls + 2 teddy bears = 10 toys in total. The answer is \\boxed{10}"},

    {"role": "user", "content": "Apply the distributive property to create an equivalent expression. 5(3p - q - 2r) = ?"},
    {"role": "assistant", "content": "The distributive property distributes the factor outside the parentheses to each term inside. So, 5 times 3p is 15p, 5 times -q is -5q, and 5 times -2r is -10r, resulting in the equivalent expression 15p - 5q - 10r. The answer is \\boxed{15p - 5q - 10r}"},

    {"role": "user", "content": "Multiply. 324 * 21"},
    {"role": "assistant", "content": "To multiply 324 by 21. Multiply 324 by 1 (the units digit of 21): 324 * 1 = 324. Multiply 324 by 20 (the tens digit of 21): 324 * 20 = 6480. Add the results from steps 1 and 2 to get the final product: 324 + 6480 = 6804. Therefore, 324 multiplied by 21 equals 6804. The answer is \\boxed{6804}"},

    {"role": "user", "content": "Solve the equation. 6 = (x/4) x = ?"},
    {"role": "assistant", "content": "To solve for x, we multiply both sides of the equation by 4 to isolate x. This gives us x = 6 * 4, which simplifies to x = 24. The answer is \\boxed{24}"},

    {"role": "user", "content": "Write your answer as a whole number and remainder. 61 / 7 = ?"},
    {"role": "assistant", "content": "When dividing 61 by 7, 7 goes into 61 8 times with a remainder of 5. The answer is \\boxed{8 R 5}"},

    {"role": "user", "content": "The bakery sold ((3)/(5))  of its cakes in the morning and ((1)/(4)) of the remaining cakes in the afternoon. How many cakes were sold altogether if there were initially 60 cakes?"},
    {"role": "assistant", "content": "The bakery initially sold 3/5 * 60 = 36 cakes in the morning. Then, it sold 1/4 * (60 - 36) = 6 cakes in the afternoon. Adding these together gives us a total of 36 + 6 = 42 cakes sold. The answer is \\boxed{42}"},

    {"role": "user", "content": "David rents a car for a day. The rental company charges an initial fee of 50 dollars plus 20 dollars for each hour the car is rented. If David pays 130 dollars for the rental, write a function C in dollars, where C(h) represents the total cost of renting the car based on the number of hours h it is rented."},
    {"role": "assistant", "content": "The total cost C consists of the initial fee of 50 dollars plus 20 dollars for each hour h the car is rented. Thus, the function C(h) = 50 + 20h. The answer is \\boxed{50 + 20h}"}
]

COT_ANSWER_ICL = [
    {"role": "user", "content": "((5)/(9)) / ((8)/(5)) = ?"},
    {"role": "assistant", "content": "To divide fractions, we can multiply the first fraction by the reciprocal of the second fraction. So, (5/9) divided by (8/5) becomes (5/9) multiplied by (5/8), which simplifies to (25/72). The answer is 25/72."},

    {"role": "user", "content": "There are 3 toy cars, 5 dolls, and 2 teddy bears on display. How many toys are there in total?"},
    {"role": "assistant", "content": "To find the total number of toys, we add the quantities of each type of toy together: 3 toy cars + 5 dolls + 2 teddy bears = 10 toys in total. The answer is 10."},

    {"role": "user", "content": "Apply the distributive property to create an equivalent expression. 5(3p - q - 2r) = ?"},
    {"role": "assistant", "content": "The distributive property distributes the factor outside the parentheses to each term inside. So, 5 times 3p is 15p, 5 times -q is -5q, and 5 times -2r is -10r, resulting in the equivalent expression 15p - 5q - 10r. The answer is 15p - 5q - 10r"},

    {"role": "user", "content": "Multiply. 324 * 21"},
    {"role": "assistant", "content": "To multiply 324 by 21. Multiply 324 by 1 (the units digit of 21): 324 * 1 = 324. Multiply 324 by 20 (the tens digit of 21): 324 * 20 = 6480. Add the results from steps 1 and 2 to get the final product: 324 + 6480 = 6804. Therefore, 324 multiplied by 21 equals 6804. The answer is 6804."},

    {"role": "user", "content": "Solve the equation. 6 = (x/4) x = ?"},
    {"role": "assistant", "content": "To solve for x, we multiply both sides of the equation by 4 to isolate x. This gives us x = 6 * 4, which simplifies to x = 24. The answer is 24."},

    {"role": "user", "content": "Write your answer as a whole number and remainder. 61 / 7 = ?"},
    {"role": "assistant", "content": "When dividing 61 by 7, 7 goes into 61 8 times with a remainder of 5. The answer is 8 R 5"},

    {"role": "user", "content": "The bakery sold ((3)/(5))  of its cakes in the morning and ((1)/(4)) of the remaining cakes in the afternoon. How many cakes were sold altogether if there were initially 60 cakes?"},
    {"role": "assistant", "content": "The bakery initially sold 3/5 * 60 = 36 cakes in the morning. Then, it sold 1/4 * (60 - 36) = 6 cakes in the afternoon. Adding these together gives us a total of 36 + 6 = 42 cakes sold. The answer is 42."},

    {"role": "user", "content": "David rents a car for a day. The rental company charges an initial fee of 50 dollars plus 20 dollars for each hour the car is rented. If David pays 130 dollars for the rental, write a function C in dollars, where C(h) represents the total cost of renting the car based on the number of hours h it is rented."},
    {"role": "assistant", "content": "The total cost C consists of the initial fee of 50 dollars plus 20 dollars for each hour h the car is rented. Thus, the function C(h) = 50 + 20h. The answer is 50 + 20h."}
]