import json
import re

import fastmcp
import mcp
import ollama

from problem_space.tasks import game24


INSTRUCTIONS_PROMPT = """INSTRUCTIONS:
You are professional in reasoning through available tools.
You structure your thoughts in the problem space.

You heavily rely on available tools, your EVERY step should contain tool calls.

- Be verbose about what are you doing
- ALWAYS wrap your final answer with tags <answer>YOUR ANSWER</answer>
"""


async def run(
    client: fastmcp.Client,
    task: game24.Task,
    max_iter: int = 200,
    model: str = 'cogito:14b',
    temperature: float = 0.7,
    seed: int = 0,
) -> tuple[str, list[dict[str, str]]]:
    available_tools = []

    res = await client.list_tools_mcp()
    for tool in res.tools:
        available_tools.append(ollama.Tool.model_validate({
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.inputSchema,
            }
        }))

    answer = "no answer"
    messages = [
        {'role': 'system', 'content': INSTRUCTIONS_PROMPT},
        {'role': 'user', 'content': task.get_prompt()},
    ]
    for i in range(max_iter):
        print(f"[{i}/{max_iter}]")

        response: ollama.ChatResponse = ollama.chat(
            model,
            messages=messages,
            tools=available_tools,
            options={
                'temperature': temperature,
                # 'repeat_penalty': 1.1,
                'num_predict': 4048,
                # 'num_ctx': 8096,
                'seed': seed,
                # 'top_k': 90,
                # 'top_p': 0.99,
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

        any_tool_failed = False
        if response.message.tool_calls is None or len(response.message.tool_calls) == 0:
            # messages.append({'role': 'system', 'content': "you MUST call a tool"})
            # print(messages[-1])
            # messages.append({'role': 'user', 'content': 'continue'})
            # print(messages[-1])
            messages.append({'role': 'user', 'content': 'continue reasoning'})
            print(messages[-1])
            continue

        for tool in response.message.tool_calls:
            try:
                output = await client.call_tool(tool.function.name, dict(tool.function.arguments))
                if output and not isinstance(output[0], mcp.types.TextContent):
                    raise ValueError(f'cannot parse tool response: {str(output)}')

                messages.append({
                    'role': 'tool',
                    'content': json.dumps({
                        'function_name': tool.function.name,
                        'request': tool.function.arguments,
                        'response': output[0].text if output else '',
                    }),
                })
                print(messages[-1])
            except Exception as e:
                any_tool_failed = True
                messages.append({'role': 'tool', 'content': f"{tool.function.name} call failed: {str(e)}"})
                print(messages[-1])

        #if any_tool_failed:
        #    messages.append({'role': 'user', 'content': "fix TOOL CALL FAILED. Don't call `reset_problem_space`, focus on initial problem"})
        #    print(messages[-1])
        # else:
        #     messages.append({'role': 'user', 'content': 'continue reasoning'})
        #     print(messages[-1])

    return answer, messages
