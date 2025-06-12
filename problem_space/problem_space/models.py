from pydantic import BaseModel, Field


class CognitiveOperator(BaseModel):
    id: int = Field(description="Operator unique ID")
    description: str = Field(description="Human-readable operator description")


class CognitiveState(BaseModel):
    id: int = Field(description="Cognitive state unique ID")
    description: str = Field(description="Human-readable state description")
    distance_to_goal: float = Field(description="Number representing estimated distance to goal")


class Transition(BaseModel):
    from_state_id: int = Field(description="ID of state to transition from")
    to_state_id: int = Field(description="ID of state to transition to")
    operator_id: int = Field(description="ID of operator to use")


class CognitiveMap(BaseModel):
    goal_description: str
    states: list[CognitiveState]
    operators: list[CognitiveOperator]
    applied_actions: list[Transition]


class StateAdded(BaseModel):
    id: int = Field(description="Cognitive state unique ID")
    distance_to_goal: float = Field(description="Number representing estimated distance to goal")


class OperatorAdded(BaseModel):
    id: int = Field(description="Cognitive operator unique ID")
