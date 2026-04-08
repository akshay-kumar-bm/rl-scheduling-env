"""Root-level grader entry points for OpenEnv judge.

Each function receives the episode trajectory and returns a float in [0.0, 1.0].
These are referenced in openenv.yaml and must be importable from the repo root.
"""

from __future__ import annotations

from typing import Any, Dict, List

from server.graders import SchedulingGrader
from server.scheduling_env_environment import SchedulingEnvironment

_grader = SchedulingGrader()
_env = SchedulingEnvironment()


def _grade_task(task_id: str, trajectory: Any = None) -> float:
    """Run a task with the environment's own state and grade it.

    If a trajectory dict is provided with 'final_state' and 'final_observation',
    those are used directly. Otherwise the grader returns 0.0.
    """
    if trajectory is not None:
        final_state = trajectory.get("final_state")
        final_obs = trajectory.get("final_observation")
        if final_state and final_obs:
            return _grader.grade_episode(final_state, final_obs)

    # Fallback: use the environment's current state
    state = _env.state
    if state.completed:
        from models import SchedulingObservation

        obs = SchedulingObservation(
            success=True,
            done=True,
            reward=state.final_reward,
        )
        return _grader.grade_episode(state, obs)

    return 0.0


def task1_easy_grader(trajectory: Any = None) -> float:
    """Grader for task1_easy: 2 attendees, free slot exists. Expected: 0.8-1.0."""
    return _grade_task("task1_easy", trajectory)


def task2_medium_grader(trajectory: Any = None) -> float:
    """Grader for task2_medium: 4 attendees, requires rescheduling. Expected: 0.5-0.8."""
    return _grade_task("task2_medium", trajectory)


def task3_hard_grader(trajectory: Any = None) -> float:
    """Grader for task3_hard: 6 attendees, cascading conflicts. Expected: 0.2-0.6."""
    return _grade_task("task3_hard", trajectory)
