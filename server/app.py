# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Meeting Scheduling RL Environment.

Uses OpenEnv's create_app() for standard routes + Gradio web UI,
then overrides /reset and /step with stateful (singleton) versions
so that HTTP-based interaction (curl, inference scripts) works correctly
across multiple calls within the same episode.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import Body
from openenv.core.env_server.http_server import create_app

try:
    from ..models import SchedulingAction, SchedulingObservation
    from .scheduling_env_environment import SchedulingEnvironment
except (ModuleNotFoundError, ImportError):
    from models import SchedulingAction, SchedulingObservation
    from server.scheduling_env_environment import SchedulingEnvironment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Create base app via OpenEnv (provides Gradio UI, WebSocket, schema, health)
# ---------------------------------------------------------------------------
app = create_app(
    env=SchedulingEnvironment,
    action_cls=SchedulingAction,
    observation_cls=SchedulingObservation,
    env_name="scheduling_env",
    max_concurrent_envs=1,
)

# ---------------------------------------------------------------------------
# Remove default stateless /reset and /step routes so we can replace them
# with stateful singleton-backed versions for HTTP interaction.
# ---------------------------------------------------------------------------
_routes_to_remove = {"/reset", "/step"}
app.routes[:] = [r for r in app.routes if getattr(r, "path", None) not in _routes_to_remove]

# ---------------------------------------------------------------------------
# Singleton environment for stateful HTTP endpoints
# ---------------------------------------------------------------------------
_env: SchedulingEnvironment = SchedulingEnvironment()


@app.post("/reset")
async def reset_handler(
    body: Optional[Dict[str, Any]] = Body(default=None),
) -> Dict[str, Any]:
    """Reset the environment to a new episode."""
    body = body or {}
    task_id = body.get("task_id", "task1_easy")

    loop = asyncio.get_event_loop()
    observation = await loop.run_in_executor(
        None, lambda: _env.reset(task_id=task_id)
    )

    obs_dict = (
        observation.model_dump()
        if hasattr(observation, "model_dump")
        else observation.__dict__
    )
    return {
        "observation": obs_dict,
        "done": getattr(observation, "done", False),
        "reward": getattr(observation, "reward", 0.0),
        **obs_dict,
    }


@app.post("/step")
async def step_handler(
    body: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    """Execute an action and return the resulting observation."""
    # Support both {"action": {...}} and direct action fields
    action_data = body.get("action", body)

    try:
        action = SchedulingAction(**action_data)
    except Exception as e:
        logger.error("Failed to deserialize action: %s", e)
        return {
            "observation": {
                "success": False,
                "error_message": f"Invalid action: {e}",
                "done": False,
                "reward": -1.0,
            },
            "done": False,
            "reward": -1.0,
        }

    loop = asyncio.get_event_loop()
    observation = await loop.run_in_executor(None, _env.step, action)

    obs_dict = (
        observation.model_dump()
        if hasattr(observation, "model_dump")
        else observation.__dict__
    )
    return {
        "observation": obs_dict,
        "done": getattr(observation, "done", False),
        "reward": getattr(observation, "reward", 0.0),
        **obs_dict,
    }


@app.get("/state")
async def state_handler() -> Dict[str, Any]:
    """Return the current internal environment state."""
    state = _env.state
    return (
        state.model_dump()
        if hasattr(state, "model_dump")
        else state.__dict__
    )


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
