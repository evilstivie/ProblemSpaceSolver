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

        response_text = ""
        for part in ollama.chat(
            model,
            messages=messages,
            options={
                'temperature': temperature,
                'repeat_penalty': 1.1,
                'top_p': 0.7,
                'top_k': 50,
            },
            stream=True,
        ):
            if not part.message.content:
                break

            response_text += part.message.content or ''
            print(part.message.content, end='', flush=True)
            if len(response_text) > 10000:
                response_text += "<interrupted>"
                break

        messages.append({'role': 'assistant', 'content': response_text})

        if not response_text:
            break

        matches = re.findall(r'<answer>(.*?)<\/answer>', response_text, re.DOTALL)
        if len(matches) > 0:
            answer = matches[0]
            break

        messages.append({'role': 'user', 'content': 'continue reasoning'})
        print(messages[-1])

    return answer, messages
