from typing import Annotated

import fastmcp
from pydantic import Field
import sympy

from . import models, registry

mcp = fastmcp.FastMCP(name="Problem Space Map")

REGISTRY = registry.ProblemSpaceRegistry()


@mcp.tool()
def start_solving_problem(
    task_description: Annotated[str, Field(description="Full task description with success criteria and complete set of constraints. May be long")],
) -> None:
    """
    You MUST first call the `start_solving_problem` tool to set a new task.
    This is crucial to build a correct problem space map and estimate distance to the goal.
    You MUST heavily rely on problem space for reasoning. Your final answer should fully satisfy the goal.

    CRITICAL: this tool should be called only once.
    """
    REGISTRY.reset(task_description)


# @mcp.tool()
# def get_operator(id: int) -> models.Cognitiveoperator:
#     """
#     operator is an ACTION which can be performed on states in problem-space
#     """
#     return REGISTRY.get_map().operators[id]


@mcp.tool()
def add_operator(
    description: Annotated[str, Field(description="Concise operator meaning. MUST contain a verb")],
    complexity: Annotated[int, Field(description="Measure of how this operator would complicate the answer")]
) -> models.OperatorAdded:
    """
    Operator is an action which can be performed on states in problem-space.

    Useful to:
    - store possible actions to then select from them using `get_insight`
    - first add more high-level operators and only after that add more specific and more problem-specific.

    RETURNS: new operator ID. You can further use this operator ID in `add_transition`

    EXAMPLES:
    - args: {"description": "swap numbers", "complexity": 1}
      returns: {"id": 42}
    - args: {"description": "put +", "complexity": 1}
      returns: {"id": 12}
    - args: {"description": "put /", "complexity": 10}
      returns: {"id": 12}
    - args: {"description": "rotate picture", "complexity": 5}
      returns: {"id": 48}

    ERRORS:
    - goal is not set, this method is called before `start_solving_problem`
    """
    return REGISTRY.add_operator(description, complexity)


# @mcp.tool()
# def get_state(id: int) -> models.CognitiveState:
#     """
#     State is a position in problem-space
#     """
#     return REGISTRY.get_map().states[id]


@mcp.tool()
def add_transition(
    from_state_id: Annotated[int, Field(description="ID of state from ProblemSpaceMap which should be previously created with `add_transition` or 0")],
    operator_id: Annotated[int, Field(description="ID of operator from ProblemSpaceMap which should be previously created with `add_operator`")],
    new_state_description: Annotated[str, Field(description="Concise new state meaning")],
) -> models.StateAdded:
    """
    State is a position in problem-space. The transition encodes a formally valid shift from one state to a NEW state using operator.

    Useful to:
    - explore new states.
    - apply newly created operators to previously discovered states.
    - store your position in a problem-space to then see a full picture using `get_insight`.
    - first add more simple states which encode full picture and only then add more specific using more complex operators.

    Requirements:
    - operator should be applicable to the state from which you make transition.
    - result of application of the operator to starting state MUST be exactly the new state.
    - don't make repeating transitions, analyze big picture with `get_insight`
    - use IDs which are returned from `add_operator` or `add_transition` or `get_insight`. Take EXACTLY ONE state from map and operator and pass their EXACT IDs to create transition to a new state.

    Distance:
        You MUST make decisions based on returned `distance_to_goal`. Low distance means you need make a small transition, slightly changing state. Big distance means you likely need make a big transition, consider new operator or another state with lower distance. Distance is estimate, not precise, you should consider higher distance to make progress if you stuck.

    RETURNS: new state ID and estimated distance to goal. You can further use this new state ID in `add_transition`

    EXAMPLES:
    - args: {"new_state_description": "5 rocks on the left, 10 rocks on the right", "from_state_id": 10, "operator_id": 0}
      returns: {"id": 40, "distance_to_goal": 14}
    - args: {"new_state_description": "hanoi disks: [A,B] [C] []", "from_state_id": 2, "operator_id": 20}
      returns: {"id": 2, "distance_to_goal": 70}

    ERRORS:
    - goal is not set, this method is called before `start_solving_problem`
    - from_state_id does not exist in problem space
    - operator_id does not exist in problem space
    - state with provided `new_state_description` already exists
    - operator can't be applied to from_state
    - result of application of operator to from_state is not equivalent to new_state
    """
    return REGISTRY.add_transition(from_state_id, operator_id, new_state_description)


@mcp.tool()
def get_insight() -> models.ProblemSpaceMap:
    """
    Get Map of your task progress with distances to goals. Carefully analyze the `ProblemSpaceMap` returned by the tool.

    Distance:
        You MUST make decisions based on `distance_to_goal` of states. Low distance means you need make a small transition, slightly changing state. Big distance means you need make a big transition, consider new operator or another state with lower distance. Distance is estimate, not precise, you should consider higher distance to make progress if you stuck.

    Useful when:
    - you think that there is no solution
    - you want to check your goal
    - you want to overview directions you have visited
    - you want to change directions
    - you have an error from `add_transition` or `add_operator` and need an ID to use

    RETURNS: full problem-space map with objects you previously saved. Last item in `transition_history` represents your current state. If you are making repeating transitions, try to start from another state based on Distance.

    HINT: If you encounter the same distance or states multiple times, try take a different directions using `get_insight`.

    EXAMPLES:
    - args: {}
      returns: {"goal_description":"Use numbers 4 4 6 8 and basic arithmetic operations (+ - * /) to obtain 24","states":[{"id":0,"description":"start","distance_to_goal":100.0}],"operators":[{"id":0,"description":"put numbers in some order"},{"id":1,"description":"put +"},{"id":2,"description":"add *"},{"id":3,"description":"add /"},{"id":4,"description":"add brackets"},{"id":5,"description":"reorder numbers"}],"transition_history":[]}

    ERRORS:
    - goal is not set, this method is called before `start_solving_problem`
    """
    return REGISTRY.get_map()


if __name__ == "__main__":
    mcp.run()
