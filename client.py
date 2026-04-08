# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scheduling Environment Client."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SchedulingAction, SchedulingObservation, SchedulingState


class SchedulingEnv(
    EnvClient[SchedulingAction, SchedulingObservation, SchedulingState]
):
    """Client for the Meeting Scheduling RL Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example::

        with SchedulingEnv(base_url="http://localhost:8000") as client:
            result = client.reset(task_id="task1_easy")
            obs = result.observation
            result = client.step(SchedulingAction(
                action_type="propose_slot",
                proposed_start="2025-04-07T10:00:00+00:00",
                proposed_duration=30,
            ))
    """

    def _step_payload(self, action: SchedulingAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[SchedulingObservation]:
        obs_data = payload.get("observation", payload)
        observation = SchedulingObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict) -> SchedulingState:
        return SchedulingState(**payload)
