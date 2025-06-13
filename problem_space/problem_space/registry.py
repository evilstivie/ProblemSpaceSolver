import ollama
from pydantic import BaseModel

from . import models


DISTANCE_EVAL_INSTRUCTIONS = """
INSTRUCTIONS:
You are a meticulous cartographer of a problem space. Your sole function is to estimate distance to goal.

The distance means conceptual distance, not necessarily a numerical distance.

A lower number means it's closer to the `goal_description`.

Be very strict at checking distance is 0. This should ONLY mean that goal is achieved, e.g. state equivalent to goal.

Max distance is 100.

EXAMPLES:
- goal is to connect nine dots arranged in a 3x3 grid using four straight lines, without lifting your pen and without retracing any lines (9-dot-problem)
  state "angles connected with line" has bigger distance than "centers of near sides connected" because second state will break the limit: Many people get stuck because they subconsciously limit themselves to drawing lines within the boundaries of the square formed by the dots

- goal is to Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
  state that uses new number ordering may be closer to goal than a lot of existing states with the same number ordering.
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
                    description="empty state",
                    distance_to_goal=100,
                )
            ],
            operators=[],
            applied_actions=[],
        )
        self.history = []

    def _evaluate_distance_with_llm(self, item_to_evaluate: str) -> float:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `create_problem_space_map` first")

        distances = []
        for _ in range(3):
            class Answer(BaseModel):
                distance: float

            messages = [
                {
                    'role': 'system',
                    'content': DISTANCE_EVAL_INSTRUCTIONS,
                },
                {
                    'role': 'user',
                    'content': f"State to estimate: {item_to_evaluate}\nGoal: {self.m.goal_description}\nFull map:\n{self.m.model_dump_json(indent=2)}",
                },
            ]

            response = ollama.chat(
                model='cogito:14b',
                messages=messages,
                format=Answer.model_json_schema(),
                options={
                    'temperature': 0.0,
                    'num_predict': 512,
                },
            )
            distances.append(Answer.model_validate_json(response.message.content or '').distance)

        distances.sort()

        return distances[1]

    def add_state(self, description: str) -> models.StateAdded:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `create_problem_space_map` first")

        for state in self.m.states:
            if state.description == description:
                raise ValueError(f"state with `description`=\"{description}\" already exists and has ID = {state.id}")

        state_id = len(self.m.states)
        distance = self._evaluate_distance_with_llm(description)
        self.m.states.append(models.CognitiveState(
            id=state_id,
            description=description,
            distance_to_goal=distance,
        ))
        return models.StateAdded(
            id=state_id,
            distance_to_goal=distance,
        )

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
