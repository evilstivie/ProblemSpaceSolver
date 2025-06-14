import json
import re

import fastmcp
import mcp
import ollama

from problem_space.tasks import game24


INSTRUCTIONS_PROMPT = """INSTRUCTIONS:
You are professional in reasoning through available tools.
You heavily rely on available tools, your EVERY step should contain tool calls.
Before solving a problem you form strict success criteria and complete set of constraints.
Your reasoning contain ONLY tool calls, no other text.

- Be verbose about what are you doing
- ALWAYS wrap your final answer with tags <answer>YOUR ANSWER</answer>. Before giving answer you always verify it against criteria.
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

        num_empty = 0
        response_text = ""
        tool_calls = []
        for part in ollama.chat(
            model,
            messages=messages,
            tools=available_tools,
            options={
                'temperature': temperature,
                # 'repeat_penalty': 1.1,
                'num_predict': 32000,
                # 'num_ctx': 8096,
                'seed': seed,
                # 'top_k': 90,
                # 'top_p': 0.99,
            },
            stream=True,
        ):
            if part.message.content is None and not part.message.tool_calls:
                num_empty += 1
                if num_empty == 5:
                    break
                continue

            response_text += part.message.content or ''
            print(part.message.content, end='', flush=True)
            if part.message.tool_calls is not None and part.message.tool_calls:
                print(json.dumps([tool.model_dump() for tool in part.message.tool_calls]), end='', flush=True)
                tool_calls.extend(part.message.tool_calls)

            if len(response_text) > 32000 or len(tool_calls) > 100:
                response_text += "<interrupted>"
                break

        messages.append({'role': 'assistant', 'content': response_text, 'tool_calls': [tool_call.model_dump() for tool_call in tool_calls]})

        if not response_text and not tool_calls:
            break

        any_tool_failed = False
        if len(tool_calls) == 0:
            matches = re.findall(r'<answer>(.*?)<\/answer>', response_text, re.DOTALL)
            if len(matches) > 0:
                answer = matches[-1]
                break

            # messages.append({'role': 'system', 'content': "you MUST call a tool"})
            # print(messages[-1])
            # messages.append({'role': 'user', 'content': 'continue'})
            # print(messages[-1])
            messages.append({'role': 'system', 'content': 'continue'})
            print(messages[-1])
            continue

        for tool in tool_calls:
            try:
                output = await client.call_tool(tool.function.name, dict(tool.function.arguments))
                if output and not isinstance(output[0], mcp.types.TextContent):
                    raise ValueError(f'cannot parse tool response: {str(output)}')

                messages.append({
                    'role': 'tool',
                    'content': json.dumps({
                        'function_name': tool.function.name,
                        'request': tool.function.arguments,
                        'response': json.loads(output[0].text) if output else '',
                    }),
                    'name': tool.function.name,
                })
                print(messages[-1])
            except Exception as e:
                any_tool_failed = True
                messages.append({'role': 'tool', 'content': f"{tool.function.name} call failed: {str(e)}", 'name': tool.function.name})
                print(messages[-1])

        messages.append({'role': 'system', 'content': 'continue, use tool responses'})
        print(messages[-1])

        #if any_tool_failed:
        #    messages.append({'role': 'user', 'content': "fix TOOL CALL FAILED. Don't call `reset_problem_space`, focus on initial problem"})
        #    print(messages[-1])
        # else:
        #     messages.append({'role': 'user', 'content': 'continue reasoning'})
        #     print(messages[-1])

    return answer, messages
