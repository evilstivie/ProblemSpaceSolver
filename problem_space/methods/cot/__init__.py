import re

import ollama

from problem_space.tasks import game24


INSTRUCTIONS_PROMPT = """INSTRUCTIONS:
You are professional in reasoning step by step.
- Be verbose about what are you doing
- ALWAYS wrap your final answer with tags <answer>YOUR ANSWER</answer>
"""


async def run(
    task: game24.Task,
    max_iter: int = 200,
    model: str = 'cogito:14b',
    temperature: float = 0.7,
) -> tuple[str, list[dict[str, str]]]:
    answer = "no answer"
    messages = [
        {'role': 'system', 'content': INSTRUCTIONS_PROMPT},
        {'role': 'user', 'content': task.get_prompt()},
    ]
    for _ in range(max_iter):
        response: ollama.ChatResponse = ollama.chat(
            model,
            messages=messages,
            options={
                'temperature': temperature,
                # 'repeat_penalty': 1.1,
                # 'num_predict': 512,
            },
        )
        messages.append(response.message.model_dump())
        print(messages[-1])

        if response.message.content is None:
            break

        matches = re.findall(r'<answer>(.*?)<\/answer>', response.message.content, re.DOTALL)
        if len(matches) > 0:
            answer = matches[0]
            break

        messages.append({'role': 'user', 'content': 'continue reasoning'})
        print(messages[-1])

    return answer, messages
