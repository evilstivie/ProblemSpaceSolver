from typing import Annotated

import fastmcp
from pydantic import Field

from . import models, registry

mcp = fastmcp.FastMCP(name="CognitiveMap")

REGISTRY = registry.ProblemSpaceRegistry()


@mcp.tool()
def create_problem_space_map(
    task_description: Annotated[str, Field(description="Full task description with success criteria and constraints")],
) -> None:
    """
    You MUST first call the `create_problem_space_map` tool to set a new task.
    This is crucial to build a correct problem space map and estimate distance to the goal.

    You MUST heavily rely on problem space for reasoning
    """
    REGISTRY.reset(task_description)


# @mcp.tool()
# def get_operator(id: int) -> models.CognitiveOperator:
#     """
#     Operator is an ACTION which can be performed on states in problem-space
#     """
#     return REGISTRY.get_map().operators[id]


@mcp.tool()
def add_operator(
    description: Annotated[str, Field(description="Concise operator meaning. MUST contain a verb")],
) -> models.OperatorAdded:
    """
    Operator is a HEURISTIC which can be performed on states in problem-space.
    FIRST add more simple heuristics like "do X for all" and only then add more specific and more low-level like "change X to Y at position Z"

    You should call `add_transition` to explore new heuristics.

    If map does not contain action you can perform, add it with tool.
    RETURNS: new operator ID. You can further use this operator ID in `add_transition`

    EXAMPLES:
    - args: {"description": "swap numbers"}
      returns: {"id": 42}
    - args: {"description": "put + operator"}
      returns: {"id": 12}
    - args: {"description": "rotate picture"}
      returns: {"id": 48}

    ERRORS:
    - goal is not set, this method is called before `create_problem_space_map`
    """
    return REGISTRY.add_operator(description)


# @mcp.tool()
# def get_state(id: int) -> models.CognitiveState:
#     """
#     State is a position in problem-space
#     """
#     return REGISTRY.get_map().states[id]


@mcp.tool()
def add_transition(
    from_state_id: Annotated[int, Field(description="ID of state from CognitiveMap which should be previously created with `add_transition` or 0")],
    operator_id: Annotated[int, Field(description="ID of operator from CognitiveMap which should be previously created with `add_operator`")],
    new_state_description: Annotated[str, Field(description="Concise new state meaning")],
) -> models.StateAdded:
    """
    State is a position in problem-space. The transition encodes a VALID shift from one state to another using operator.
    FIRST add more simple states which encode full picture and only then add more specific.

    You should call `add_transition` to explore new states.

    IMPORTANT: use IDs which are returned from `add_operator` or `add_transition` or `get_problem_space_map`. Take EXACTLY ONE state from map and operator and pass their EXACT IDs to create transition to a new state.

    RETURNS: new state ID and estimated distance to goal
    You MUST make decisions based on `distance_to_goal`. Distance is estimate, not precise, distance of 0.0 may mean that you need one additional small step.
    You can further use this new state ID in `add_transition`

    EXAMPLES:
    - args: {"new_state_description": "5 rocks on the left, 10 rocks on the right", "from_state_id": 10, "operator_id": 0}
      returns: {"id": 40, "distance_to_goal": 14}
    - args: {"new_state_description": "hanoi disks: [A,B] [C] []", "from_state_id": 2, "operator_id": 20}
      returns: {"id": 2, "distance_to_goal": 70}

    HINT: If you encounter the same distance or states multiple times, try take a different directions using `get_problem_space_map`.

    ERRORS:
    - goal is not set, this method is called before `create_problem_space_map`
    - from_state_id does not exist in problem space
    - operator_id does not exist in problem space
    """
    return REGISTRY.add_transition(from_state_id, operator_id, new_state_description)


@mcp.tool()
def get_problem_space_map() -> models.CognitiveMap:
    """
    Problem space maps your task progress. Get cognitive map with distances to goals.
    Carefully analyze the `CognitiveMap` returned by the tool. You MUST make decisions based on `distance_to_goal` in observed map. Distance is an estimate, not precise, distance of 0.0 may mean that you need one additional small step.

    Useful when:
    - you think that there is no solution
    - you want to check your goal
    - you want to overview directions you have visited
    - you want to change directions

    HINT: If you encounter the same distance or states multiple times, try take a different directions using `get_problem_space_map`.

    EXAMPLES:
    - args: {}
      returns: {"goal_description":"Use numbers 4 4 6 8 and basic arithmetic operations (+ - * /) to obtain 24","states":[{"id":0,"description":"start","distance_to_goal":100.0}],"operators":[{"id":0,"description":"put numbers in some order"},{"id":1,"description":"put +"},{"id":2,"description":"add *"},{"id":3,"description":"add /"},{"id":4,"description":"add brackets"},{"id":5,"description":"reorder numbers"}],"applied_actions":[]}

    You can further use this IDs from map in `add_transition`

    ERRORS:
    - goal is not set, this method is called before `create_problem_space_map`
    """
    return REGISTRY.get_map()


if __name__ == "__main__":
    mcp.run()
