import ollama
from pydantic import BaseModel

from . import models


DISTANCE_EVAL_INSTRUCTIONS = """INSTRUCTIONS:
1. Your sole function is to estimate the distance of a 'New state' from a 'Target state'. A lower number (minimum 0) is better. Max distance is 100.
2. Be very strict at checking RULES provided in goal statement.
3. Provide ONLY the final estimated distance as a single number.
"""

#  For example, if previous distance is 100, new distance will unlikely be 10.



class ProblemSpaceRegistry:
    def __init__(self):
        self.reset("unknown")

    def reset(self, goal: str):
        if m := getattr(self, 'm', None):
            if m.goal_description != "unknown":
                error_message = (
                    f"Error: Goal is already set an equals to '{m.goal_description}'."
                    "You are likely exploring in a circle. "
                    "Suggestion: Get current problem space using `get_insight` call."
                )
                raise ValueError(error_message)

        self.m = models.ProblemSpaceMap(
            goal_description=goal,
            states=[
                models.State(
                    id=0,
                    description="nothing",
                    distance_to_goal=100,
                )
            ],
            operators=[],
            transition_history=[],
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
            raise ValueError("goal is unknown, call `start_solving_problem` first")

        class Answer(BaseModel):
            distance: float

        messages = [
            {
                'role': 'system',
                'content': DISTANCE_EVAL_INSTRUCTIONS,
            },
            {
                'role': 'user',
                'content': f"""Target state:
"{self.m.goal_description}"

Previous state (previous distance = {previous_distance}):
"{previous_state}"

New state (distance = ?):
"{new_state}"
"""
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
        return Answer.model_validate_json(response.message.content or '').distance

    def add_operator(self, description: str, complexity: int) -> models.OperatorAdded:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `start_solving_problem` first")

        for operator in self.m.operators:
            if operator.description == description:
                # return models.OperatorAlreadyExistsError(existing_id=operator.id)

                error_message = (
                    f"Error: The operator '{description}' already exists with ID {operator.id}. "
                    "You are likely exploring in a circle. "
                    "Suggestion: Try applying THIS operator to a state using `add_transition`, or use `get_insight` to find a completely new path with a lower distance."
                )

                raise ValueError(error_message)

        op_id = len(self.m.operators)
        self.m.operators.append(models.Operator(
            id=op_id,
            description=description,
            complexity=complexity,
        ))
        return models.OperatorAdded(
            id=op_id,
        )

    def add_transition(self, from_state_id: int, operator_id: int, new_state_description: str) -> models.StateAdded:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `start_solving_problem` first")
        if from_state_id >= len(self.m.states):
            raise ValueError(f"Origin state {from_state_id} not found. Use only existing states. First add state with `add_transition` and use ID returned from that function call")
        if operator_id >= len(self.m.operators):
            raise ValueError(f"Operator '{operator_id}' not found. First add operator with `add_operator` and use ID returned from that function call")

        for state in self.m.states:
            if state.description == new_state_description:
                self.m.transition_history.append(
                    models.Transition(
                        from_state_id=from_state_id,
                        to_state_id=state.id,
                        operator_id=operator_id,
                        is_new=False
                    )
                )
                # return models.StateAlreadyExistsError(
                #     existing_id=state.id,
                #     distance_to_goal=state.distance_to_goal,
                # )
                error_message = (
                    f"Error: The state '{new_state_description}' already exists with ID {state.id}. "
                    "You are likely exploring in a circle. "
                    "Suggestion: Try making a DIFFERENT transition, or use `get_insight` to find a completely new path with a lower distance."
                )

                raise ValueError(error_message)
                # raise ValueError(f"state with `description`=\"{new_state_description}\" already exists and has ID = {state.id}")

        state_id = len(self.m.states)
        distance = self._evaluate_distance_with_llm(
            previous_state=self.m.states[from_state_id].description,
            previous_distance=self.m.states[from_state_id].distance_to_goal,
            operator_description=self.m.operators[operator_id].description,
            new_state=new_state_description,
        )
        state = models.State(
            id=state_id,
            description=new_state_description,
            distance_to_goal=distance,
        )
        self.m.states.append(state)

        self.m.transition_history.append(
            models.Transition(
                from_state_id=from_state_id,
                to_state_id=state.id,
                operator_id=operator_id,
                is_new=True
                # distance_delta=state.distance_to_goal-self.m.states[from_state_id].distance_to_goal,
            )
        )
        return models.StateAdded(
            id=state.id,
            distance_to_goal=state.distance_to_goal,
        )

    def get_map(self) -> models.ProblemSpaceMap:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `start_solving_problem` first")
        return self.m
