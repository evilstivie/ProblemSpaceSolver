import ollama
from pydantic import BaseModel

from . import models


DISTANCE_EVAL_INSTRUCTIONS = """
INSTRUCTIONS:
1. Your sole function is to estimate distance to goal.
2. A lower number means it's closer to the `goal_description`.
3. Distance is estimate of how the new state is more likely to reach the goal than the previous.
4. State which uses some incorrect objects which are not stated in goal SHOULD have a very big distance because it is formally incorrect.
5. Be very strict at checking distance is 0. This should ONLY mean that goal is achieved, e.g. state equivalent to goal.
6. Max distance is 100.
"""

#  For example, if previous distance is 100, new distance will unlikely be 10.



class ProblemSpaceRegistry:
    def __init__(self):
        self.reset("unknown")

    def reset(self, goal: str):
        self.m = models.ProblemSpaceMap(
            goal_description=goal,
            states=[
                models.State(
                    id=0,
                    description="nothing",
                    distance_to_goal=100,
                )
            ],
            heuristics=[],
            transition_history=[],
        )
        self.history = []

    def _evaluate_distance_with_llm(
        self,
        previous_state: str,
        previous_distance: float,
        heuristic_description: str,
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

Heuristic used:
"{heuristic_description}"

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

    def add_heuristic(self, description: str, complexity: int) -> models.HeuristicAdded | models.HeuristicAlreadyExistsError:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `start_solving_problem` first")

        for heuristic in self.m.heuristics:
            if heuristic.description == description:
                return models.HeuristicAlreadyExistsError(existing_id=heuristic.id)
                # raise ValueError(f"heuristic with `description`=\"{description}\" already exists and has ID = {heuristic.id}")

        op_id = len(self.m.heuristics)
        self.m.heuristics.append(models.Heuristic(
            id=op_id,
            description=description,
            complexity=complexity,
        ))
        return models.HeuristicAdded(
            id=op_id,
        )

    def add_transition(self, from_state_id: int, heuristic_id: int, new_state_description: str) -> models.StateAdded | models.StateAlreadyExistsError:
        if self.m.goal_description == "unknown":
            raise ValueError("goal is unknown, call `start_solving_problem` first")
        if from_state_id >= len(self.m.states):
            raise ValueError(f"Origin state {from_state_id} not found. Use only existing states. First add state with `add_transition` and use ID returned from that function call")
        if heuristic_id >= len(self.m.heuristics):
            raise ValueError(f"Heuristic '{heuristic_id}' not found. First add heuristic with `add_heuristic` and use ID returned from that function call")

        for state in self.m.states:
            if state.description == new_state_description:
                return models.StateAlreadyExistsError(
                    existing_id=state.id,
                    distance_to_goal=state.distance_to_goal,
                )

                # raise ValueError(f"state with `description`=\"{new_state_description}\" already exists and has ID = {state.id}")

        state_id = len(self.m.states)
        distance = self._evaluate_distance_with_llm(
            previous_state=self.m.states[from_state_id].description,
            previous_distance=self.m.states[from_state_id].distance_to_goal,
            heuristic_description=self.m.heuristics[heuristic_id].description,
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
                heuristic_id=heuristic_id,
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
