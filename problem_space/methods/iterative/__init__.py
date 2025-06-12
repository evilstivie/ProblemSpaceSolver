import ollama

import fastmcp

from problem_space.tasks import game24


INSTRUCTIONS_PROMPT = """INSTRUCTIONS:
You solve problems by following a strict Observe-Orient-Decide-Act loop. Your goal is to reach a state that satisfies the problem's requirements.

REMEMBER:
- DON'T WRITE MARKDOWN for your tool calls. For example isntead of writing '```json {{\"funciton\": \"reset_problem_space\"}}```' you MUST write '{{\"function\": \"reset_problem_space\"}}'
- Be verbose about what are you doing

You are professional in reasoning through cognitive space map. You structure your thoughts in the problem space.

**Your Cognitive Loop:**

**1. SETUP:** (add explicit <setup> token)
   - you MUST first call the `reset_problem_space` tool to set a new goal. This is crucial
     to build a correct problem space map and estimate distance to the goal.

**2. OBSERVE (Every Step):** (add explicit <observe> token)
   - Before every action, you MUST call `get_problem_space_map` to get the latest analysis of your progress.
     Carefully analyze the `CognitiveMap` returned by the tool. You MUST make decisions based on
     `distance_to_goal` in observed map

**3. ORIENT (The Solution Step):** (add explicit <orient> token)
   - Based on your orientation, formulate your SINGLE next atomic reasoning step.
   - According to observation do ONE of:
     * `add_operator`: if map does not contain action you can perform, add it with tool
     * `add_transition`: take EXACTLY ONE state from map and operator and pass their EXACT ids to create transition to a new state

**4. GOAL CHECK:** (add explicit <goal check> token)
   - If you believe you have reached a state that solves the problem, MOVE TO STEP 5.
   - Otherwise CONTINUE TO STEP 2.

**5. ANSWER:** (add explicit <answer> token)
   - state your answer clearly

STOP AFTER EACH STEP AND WAIT ME TO LET YOU CONTINUE. MAKE ONLY ONE TOOL CALL.

EXAMPLE OF YOUR PROCESS:
<setup>
Q: {'function': 'reset_problem_space', 'arguments': {'new_goal': 'Use numbers 4 4 6 8 and basic arithmetic operations (+ - * /) to obtain 24'}}

<observe>
Q: {'function': 'get_problem_space'}
A: {"goal_description":"Use numbers 4 4 6 8 and basic arithmetic operations (+ - * /) to obtain 24","states":[{"id":0,"description":"start","distance_to_goal":100.0}],"operators":[],"applied_actions":[]}
<orient>
Q: {'function': 'add_operator', 'arguments': {'description': 'put numbers in some order'}}
A: 0
Q: {'function': 'add_operator', 'arguments': {'description': 'put +'}}
A: 1
...
Q: {'function': 'add_operator', 'arguments': {'description': 'change number ordering'}}
A: 5
...
<observe>
Q: {'function': 'get_problem_space'}
A: {"goal_description":"Use numbers 4 4 6 8 and basic arithmetic operations (+ - * /) to obtain 24","states":[{"id":0,"description":"start","distance_to_goal":100.0}],"operators":[{"id":0,"description":"put numbers in some order"},{"id":1,"description":"put +"},{"id":2,"description":"add *"},{"id":3,"description":"add /"},{"id":4,"description":"add brackets"},{"id":5,"description":"reorder numbers"}],"applied_actions":[]}
<orient>
Q: {'function': 'add_transition', 'arguments': {'from_state_id': 0, 'operator_id': 0, 'new_state_description': '6 _ 4 _ 4 _ 8'}}
A: 0
...
<observe>
Q: {'function': 'get_problem_space_map'}
A: {"goal_description":"Use numbers 4 4 6 8 and basic arithmetic operations (+ - * /) to obtain 24","states":[{"id":0,"description":"start","distance_to_goal":100.0}, {"id":1,"description":"6 _ 4 _ 4 _ 8","distance_to_goal":50.0}],"operators":[{"id":0,"description":"put numbers in some order"},{"id":1,"description":"put +"},{"id":2,"description":"add *"},{"id":3,"description":"add /"},{"id":4,"description":"add brackets"},{"id":5,"description":"reorder numbers"}],"applied_actions":[{"from_state_id":0,"to_state_id":1,"operator_id":0}]}}
...
<answer>
(8 + 4) * (6 - 4) = 24

REMEMBER: ALWAYS add <answer> token on final step when you provide answer.
"""


async def run(client: fastmcp.Client, task: game24.Task, max_iter: int = 200, model: str = 'cogito:14b') -> str:
    res = await client.list_tools_mcp()
    available_tools = []
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
        )
        messages.append(response.message.model_dump())
        print('solver:', response.message)

        if response.message.content is None or 'answer:' in response.message.content:
            break

        any_tool_failed = False
        if response.message.tool_calls is None or len(response.message.tool_calls) == 0:
            messages.append({'role': 'user', 'content': "you MUST call a tool, focus on initial problem"})
            continue

        for tool in response.message.tool_calls:
            print(f'tool call: {tool.function.name}, {tool.function.arguments}')
            try:
                output = await client.call_tool(tool.function.name, dict(tool.function.arguments))
                print('tool output:', output)
                messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
            except Exception as e:
                any_tool_failed = True
                print('tool failed:', str(e))
                messages.append({'role': 'tool', 'content': f"TOOL CALL FAILED: {str(e)}", 'name': tool.function.name})

        if any_tool_failed:
            messages.append({'role': 'user', 'content': "call a different tool to fix TOOL CALL FAILED. Don't call `reset_problem_space`, focus on initial problem"})
        else:
            messages.append({'role': 'user', 'content': 'continue reasoning about initial problem'})

    return messages[-1]["content"]
