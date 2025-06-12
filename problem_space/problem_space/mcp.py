from fastmcp import FastMCP

from . import models, registry

mcp = FastMCP(name="CognitiveMap")

REGISTRY = registry.ProblemSpaceRegistry()


@mcp.tool()
def reset_problem_space(new_goal: str) -> None:
    """
    if goal in map does not represent your goal,

    PARAMETERS:
    - new_goal: Goal you want to achieve when solving a problem
    """
    REGISTRY.reset(new_goal)


@mcp.tool()
def add_operator(description: str) -> models.OperatorAdded:
    """
    if map does not contain action you can perform, add it with tool

    PARAMETERS:
    - description: Your current thinking step, which can include:
    * Regular analytical steps - states or operators
    * Transitions - transition from one state to another by applying an operator

    RETURNS: new operator ID and estimated potential
    """
    return REGISTRY.add_operator(description)


@mcp.tool()
def add_transition(from_state_id: int, operator_id: int, new_state_description: str) -> models.StateAdded:
    """
    take EXACTLY ONE state from map and operator and pass their EXACT ids to create transition to a new state

    PARAMETERS:
    - new_state_description: Your current thinking step, which can include:
    * Regular analytical steps - states or operators
    * Transitions - transition from one state to another by applying an operator
    - from_state_id: ID of state from CognitiveMap obtained with `get_problem_space_map`
    - operator_id: ID of operator from CognitiveMap obtained with `get_problem_space_map`

    RETURNS: new state ID and estimated distance to goal
    """
    return REGISTRY.add_transition(from_state_id, operator_id, new_state_description)


@mcp.tool()
def get_problem_space_map() -> models.CognitiveMap:
    """
    get cognitive map with distances to goals
    """
    return REGISTRY.get_map()


if __name__ == "__main__":
    mcp.run()
