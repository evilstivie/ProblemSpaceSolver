import collections.abc as cabc
import re
import os
import sympy
import pandas as pd


# 5-shot
STANDARD_PROMPT = '''RULES:
The main goal is to reach the target '24' using ALL numbers from the input set exactly once using basic arithmetic operations (+ - * /).

CRITICALLY, state is analyzed based on these rules:
    - **RULE A (GOAL ACHIEVED):** If the 'New state' is a valid mathematical expression that evaluates to 24 AND uses a plausible number of the input digits, the distance is EXACTLY 0.
    - **RULE B (NUMERICAL CLOSENESS):** If the 'New state' is a valid expression that does NOT evaluate to 24, calculate its result. The base distance should be `abs(result - 24)`.
    - **RULE C (EFFICIENCY PENALTY):** ADD A PENALTY to the distance based on how many numbers from the input set were used. A state that uses 3 numbers but is still far from 24 is WORSE than a state that uses 2 numbers and is equally far. For example, `(8*4)-6=26` (3 numbers used, distance ~2) is a worse path than `8*4=32` (2 numbers used, distance ~8). Add a penalty of `10 * (number of digits used - 2)`.
    - **RULE D (NON-NUMERIC STATES):** If the 'New state' is not a full expression (e.g., 'numbers 8 and 4 are combined'), estimate distance based on progress. A state with 2 numbers combined might be 75, 3 numbers combined might be 50. The initial state is 100.

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
