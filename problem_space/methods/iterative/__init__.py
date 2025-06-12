import json
import re

import fastmcp
import mcp
import ollama

from problem_space.tasks import game24


INSTRUCTIONS_PROMPT = """INSTRUCTIONS:
- Be verbose about what are you doing
- Use cognitive tools to reach your goal
- Make very atomic steps, use tools
- ALWAYS wrap your final answer with tags <answer>YOUR ANSWER</answer>
"""


async def run(
    client: fastmcp.Client,
    task: game24.Task,
    max_iter: int = 200,
    model: str = 'cogito:14b',
    temperature: float = 0.7,
) -> str:
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

    messages = [
        {'role': 'user', 'content': INSTRUCTIONS_PROMPT},
        {'role': 'user', 'content': task.get_prompt()},
    ]
    for _ in range(max_iter):
        response: ollama.ChatResponse = ollama.chat(
            model,
            messages=messages,
            tools=available_tools,
            options={'temperature': temperature},
        )
        messages.append(response.message.model_dump())
        print(messages[-1])

        if response.message.content is None:
            break

        matches = re.findall(r'<answer>(.*?)</answer>', response.message.content)
        if len(matches) > 0:
            return matches[0]

        any_tool_failed = False
        if response.message.tool_calls is None or len(response.message.tool_calls) == 0:
            # messages.append({'role': 'user', 'content': "you MUST call a tool, focus on initial problem"})
            # print(messages[-1])
            messages.append({'role': 'user', 'content': 'continue'})
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
                messages.append({'role': 'tool', 'content': f"TOOL CALL FAILED: {str(e)}", 'name': tool.function.name})
                print(messages[-1])

        # if any_tool_failed:
        #     messages.append({'role': 'user', 'content': "call a different tool to fix TOOL CALL FAILED. Don't call `reset_problem_space`, focus on initial problem"})
        #     print(messages[-1])
        # else:
        messages.append({'role': 'user', 'content': 'continue'})
        print(messages[-1])

    return messages[-1]["content"]
