import collections.abc as cabc
import re
import os
import sympy
import pandas as pd


# 5-shot
STANDARD_PROMPT = '''RULES:
Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
You MUST use all the numbers from Input in their original quantity.
You MUST NOT add extra numbers.
You MAY use one operator multiple times (e.g. use '+' 3 times in expression).
EACH your step must contain only numbers from Input set in their original quantity.
Solution always exists.

EXAMPLES:
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

YOUR TASK:
Input: {input}
'''


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
