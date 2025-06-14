import ollama
from pydantic import BaseModel

from . import models


DISTANCE_EVAL_INSTRUCTIONS = """
INSTRUCTIONS:
Your sole function is to estimate distance to goal.
A lower number means it's closer to the `goal_description`.

Be very strict at checking distance is 0. This should ONLY mean that goal is achieved, e.g. state equivalent to goal.

Max distance is 100. Min distance is 0.

EXAMPLE:
- For goal "Use numbers and basic arithmetic operations (+ - * /) to obtain 24." the state which is simpler and numerically closer to 24. Evaluate if given numbers can reach 24 (smaller distance means more likely) with small number of transformations.
"""


class ProblemSpaceRegistry:
    def __init__(self):
        self.reset("unknown")

    def reset(self, goal: str):
        self.m = models.CognitiveMap(
            goal_description=goal,
            states=[
                models.CognitiveState(
                    id=0,
                    description="nothing",
                    distance_to_goal=100,
                )
            ],
            operators=[],
            applied_actions=[],
        )
        self.history = []

    def _evaluate_distance_with_llm(
        self,
        previous_state: str,
        previous_distance: float,
        operator_description: str,
        new_state: str,
    ) -> float:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `create_problem_space_map` first")

        class Answer(BaseModel):
            distance: float

        messages = [
            {
                'role': 'system',
                'content': DISTANCE_EVAL_INSTRUCTIONS,
            },
            {
                'role': 'user',
                'content': f"""Previous state (distance {previous_distance}):
{previous_state}

Heuristics applied:
{operator_description}

State to estimate:
{new_state}

Original Goal for the solver:
"{self.m.goal_description}"
"""
            },
        ]

        response = ollama.chat(
            model='llama3.1:8b',
            messages=messages,
            format=Answer.model_json_schema(),
            options={
                'temperature': 0.0,
                'num_predict': 512,
            },
        )
        return Answer.model_validate_json(response.message.content or '').distance

    def add_operator(self, description: str) -> models.OperatorAdded:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `create_problem_space_map` first")

        for operator in self.m.operators:
            if operator.description == description:
                raise ValueError(f"operator with `description`=\"{description}\" already exists and has ID = {operator.id}")

        op_id = len(self.m.operators)
        self.m.operators.append(models.CognitiveOperator(
            id=op_id,
            description=description,
        ))
        return models.OperatorAdded(
            id=op_id,
        )

    def add_transition(self, from_state_id: int, operator_id: int, new_state_description: str) -> models.StateAdded:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `create_problem_space_map` first")
        if from_state_id >= len(self.m.states):
            raise ValueError(f"Origin state {from_state_id} not found. Use only existing states. First add state with `add_transition` and use ID returned from that function call")
        if operator_id >= len(self.m.operators):
            raise ValueError(f"Operator '{operator_id}' not found. First add operator with `add_operator` and use ID returned from that function call")

        for state in self.m.states:
            if state.description == new_state_description:
                raise ValueError(f"state with `description`=\"{new_state_description}\" already exists and has ID = {state.id}")

        state_id = len(self.m.states)
        distance = self._evaluate_distance_with_llm(
            previous_state=self.m.states[from_state_id].description,
            previous_distance=self.m.states[from_state_id].distance_to_goal,
            operator_description=self.m.operators[operator_id].description,
            new_state=new_state_description,
        )
        self.m.states.append(models.CognitiveState(
            id=state_id,
            description=new_state_description,
            distance_to_goal=distance,
        ))
        return models.StateAdded(
            id=state_id,
            distance_to_goal=distance,
        )

        new_state = self.add_state(new_state_description)
        self.m.applied_actions.append(
            models.Transition(
                from_state_id=from_state_id,
                to_state_id=new_state.id,
                operator_id=operator_id,
            )
        )
        return new_state

    def get_map(self) -> models.CognitiveMap:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `create_problem_space_map` first")
        return self.m
