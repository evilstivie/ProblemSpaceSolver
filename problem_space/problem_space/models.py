import typing

from pydantic import BaseModel, Field


class Heuristic(BaseModel):
    id: int = Field(description="Heuristic unique ID")
    description: str = Field(description="Human-readable heuristic description")
    complexity: int = Field(description="Measure of how this heuristic complicates the state")


class State(BaseModel):
    id: int = Field(description="State unique ID")
    description: str = Field(description="Human-readable state description")
    distance_to_goal: float = Field(description="Number representing estimated distance to goal")


class Transition(BaseModel):
    from_state_id: int = Field(description="ID of state to transition from")
    to_state_id: int = Field(description="ID of state to transition to")
    heuristic_id: int = Field(description="ID of heuristic to use")
    # distance_delta: float


class ProblemSpaceMap(BaseModel):
    goal_description: str
    states: list[State]
    heuristics: list[Heuristic]
    transition_history: list[Transition]


class StateAdded(BaseModel):
    id: int = Field(description="State unique ID")
    distance_to_goal: float = Field(description="Number representing estimated distance to goal")


class StateAlreadyExistsError(BaseModel):
    error: str = Field(default="state already exists")
    existing_id: int = Field(description="ID of state which already exists")
    distance_to_goal: float = Field(description="Number representing estimated distance to goal")


class HeuristicAdded(BaseModel):
    id: int = Field(description="Heuristic unique ID")


class HeuristicAlreadyExistsError(BaseModel):
    error: str = Field(default="heuristic already exists")
    existing_id: int = Field(description="ID of heuristic which already exists")
