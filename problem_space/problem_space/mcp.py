from fastmcp import FastMCP

from . import models, registry

mcp = FastMCP(name="CognitiveMap")

REGISTRY = registry.ProblemSpaceRegistry()


@mcp.tool()
def reset_problem_space(new_goal: str) -> None:
    """
    If goal in map does not represent your goal, set new goal.

    Problem space is very useful for creative problems.
    Use it when:
    - Breaking down complex problems into steps
    - Planning and design with room for revision
    - Analysis that might need course correction
    - Problems where the full scope might not be clear initially
    - Problems that require a multi-step solution
    - Tasks that need to maintain context over multiple steps
    - Situations where irrelevant information needs to be filtered out

    PARAMETERS:
    - new_goal: Goal you want to achieve when solving a problem
    """
    REGISTRY.reset(new_goal)


@mcp.tool()
def get_operator(id: int) -> models.CognitiveOperator:
    """
    Operator is an ACTION which can be performed on states in problem-space

    Problem space is very useful for creative problems.
    Use it when:
    - Breaking down complex problems into steps
    - Planning and design with room for revision
    - Analysis that might need course correction
    - Problems where the full scope might not be clear initially
    - Problems that require a multi-step solution
    - Tasks that need to maintain context over multiple steps
    - Situations where irrelevant information needs to be filtered out
    """
    return REGISTRY.get_map().operators[id]


@mcp.tool()
def add_operator(description: str) -> models.OperatorAdded:
    """
    If map does not contain action you can perform, add it with tool.

    Problem space is very useful for creative problems.
    Use it when:
    - Breaking down complex problems into steps
    - Planning and design with room for revision
    - Analysis that might need course correction
    - Problems where the full scope might not be clear initially
    - Problems that require a multi-step solution
    - Tasks that need to maintain context over multiple steps
    - Situations where irrelevant information needs to be filtered out

    PARAMETERS:
    - description: Concise operator meaning. MUST contain a verb

    EXAMPLES:
    {"description": "swap numbers"}
    {"description": "put + operator"}
    {"description": "rotate picture"}

    RETURNS: new operator ID
    """
    return REGISTRY.add_operator(description)


@mcp.tool()
def get_state(id: int) -> models.CognitiveState:
    """
    State is a position in problem-space
    """
    return REGISTRY.get_map().states[id]


@mcp.tool()
def add_transition(from_state_id: int, operator_id: int, new_state_description: str) -> models.StateAdded:
    """
    Take EXACTLY ONE state from map and operator and pass their EXACT ids to create transition to a new state

    Problem space is very useful for creative problems.
    Use it when:
    - Breaking down complex problems into steps
    - Planning and design with room for revision
    - Analysis that might need course correction
    - Problems where the full scope might not be clear initially
    - Problems that require a multi-step solution
    - Tasks that need to maintain context over multiple steps
    - Situations where irrelevant information needs to be filtered out

    PARAMETERS:
    - new_state_description: Concise new state meaning
    - from_state_id: ID of state from CognitiveMap obtained with `get_problem_space_map`
    - operator_id: ID of operator from CognitiveMap obtained with `get_problem_space_map`

    EXAMPLES:
    {"new_state_description": "5 rocks on the left, 10 rocks on the right", "from_state_id": 10, "operator_id": 0}
    {"new_state_description": "hanoi disks: [A,B] [C] []", "from_state_id": 2, "operator_id": 20}

    RETURNS: new state ID and estimated distance to goal
    """
    return REGISTRY.add_transition(from_state_id, operator_id, new_state_description)


@mcp.tool()
def get_problem_space_map() -> models.CognitiveMap:
    """
    Problem space maps your task progress.

    Get cognitive map with distances to goals

    Problem space is very useful for creative problems.
    Use it when:
    - Breaking down complex problems into steps
    - Planning and design with room for revision
    - Analysis that might need course correction
    - Problems where the full scope might not be clear initially
    - Problems that require a multi-step solution
    - Tasks that need to maintain context over multiple steps
    - Situations where irrelevant information needs to be filtered out
    """
    return REGISTRY.get_map()


if __name__ == "__main__":
    mcp.run()
