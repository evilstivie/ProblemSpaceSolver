import collections.abc as cabc
import re
import os
import sympy
import pandas as pd


# 5-shot
STANDARD_PROMPT = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You must use all numbers from input set. REMEMBER: you can use operator multiple times (e.g. "1 + 1 + 1 + 1"
if "Input: 1 1 1 1")
Input: 4 4 6 8
<answer>(4 + 8) * (6 - 4) = 24</answer>
Input: 2 9 10 12
<answer>2 * 12 * (10 - 9) = 24</answer>
Input: 4 9 10 13
<answer>(13 - 9) * (10 - 4) = 24</answer>
Input: 1 4 8 8
<answer>(8 / 4 + 1) * 8 = 24</answer>
Input: 5 5 5 9
<answer>5 + 5 + 5 + 9 = 24</answer>
Input: {input}
'''

# You must use all numbers from input set.
# The solution always exists.
#
# CRUCIAL: solution must contain exactly 4 numbers
#
# REMEMBER: you can use operator multiple times (e.g. "1 + 1 + 1 + 1" if "Input: 1 1 1 1")



class Task:
    def __init__(self, input: str):
        self.input = input

    def get_prompt(self) -> str:
        return STANDARD_PROMPT.format(input=self.input)

    def validate(self, answer: str) -> bool:
        expression = answer.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.input)
        if sorted(numbers) != sorted(problem_numbers):
            return False
        try:
            return sympy.simplify(sympy.parse_expr(expression)) == 24
        except Exception:
            return False



def iter_tasks() -> cabc.Iterator[Task]:
    module_dir = os.path.dirname(__file__)
    data_path = os.path.join(module_dir, 'data.csv')
    for input in pd.read_csv(data_path)['Puzzles']:
        yield Task(input)
