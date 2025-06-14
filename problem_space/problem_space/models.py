import typing

from pydantic import BaseModel, Field


class Operator(BaseModel):
    id: int = Field(description="Operator unique ID")
    description: str = Field(description="Human-readable operator description")
    complexity: int = Field(description="Measure of how this operator complicates the state")


class State(BaseModel):
    id: int = Field(description="State unique ID")
    description: str = Field(description="Human-readable state description")
    distance_to_goal: float = Field(description="Number representing estimated distance to goal")


class Transition(BaseModel):
    from_state_id: int = Field(description="ID of state to transition from")
    to_state_id: int = Field(description="ID of state to transition to")
    operator_id: int = Field(description="ID of operator to use")
    is_new: bool = Field(description="Did this transition made a new state?")
    # distance_delta: float


class ProblemSpaceMap(BaseModel):
    goal_description: str
    states: list[State]
    operators: list[Operator]
    transition_history: list[Transition]


class StateAdded(BaseModel):
    id: int = Field(description="State unique ID")
    distance_to_goal: float = Field(description="Number representing estimated distance to goal")


class OperatorAdded(BaseModel):
    id: int = Field(description="Operator unique ID")
