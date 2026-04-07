# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Meeting Scheduling RL Environment.

Uses a custom HTTP server pattern (based on calendar_env reference)
to maintain a persistent environment instance across HTTP calls,
enabling stateful multi-step episodes via /reset → /step → /state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI
from openenv.core.env_server.http_server import HTTPEnvServer

try:
    from ..models import SchedulingAction, SchedulingObservation, SchedulingState
    from .scheduling_env_environment import SchedulingEnvironment
except (ModuleNotFoundError, ImportError):
    from models import SchedulingAction, SchedulingObservation, SchedulingState
    from server.scheduling_env_environment import SchedulingEnvironment

logger = logging.getLogger(__name__)


class SchedulingHTTPEnvServer(HTTPEnvServer):
    """Custom HTTP server that maintains a persistent env instance.

    Follows the pattern from OpenEnv's calendar_env: subclass HTTPEnvServer,
    create one persistent environment, and register custom routes that
    use it for all HTTP requests.
    """

    def __init__(self, env, action_cls, observation_cls):
        self.action_cls = action_cls
        self.observation_cls = observation_cls
        super().__init__(env=env, action_cls=action_cls, observation_cls=observation_cls)

        # Persistent environment for HTTP endpoints
        if callable(self._env_factory):
            self.env = self._env_factory()
        else:
            self.env = self._env_factory

    def register_routes(self, app: FastAPI) -> None:  # type: ignore[override]
        """Register custom /reset, /step, /state endpoints."""

        @app.post("/reset")
        async def reset_handler(
            body: Optional[Dict[str, Any]] = Body(default=None),
        ) -> Dict[str, Any]:
            body = body or {}
            task_id = body.get("task_id", "task1_easy")

            loop = asyncio.get_event_loop()
            observation = await loop.run_in_executor(
                None, lambda: self.env.reset(task_id=task_id)
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
            # Support both {"action": {...}} and direct action fields
            action_data = body.get("action", body)

            try:
                action = self.action_cls(**action_data)
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
            observation = await loop.run_in_executor(
                None, self.env.step, action
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

        @app.get("/state")
        async def state_handler() -> Dict[str, Any]:
            state = self.env.state
            return (
                state.model_dump()
                if hasattr(state, "model_dump")
                else state.__dict__
            )

        @app.get("/health")
        async def health_handler() -> Dict[str, str]:
            return {"status": "healthy", "environment": "scheduling_env"}


def create_scheduling_environment():
    """Factory function for the scheduling environment."""
    return SchedulingEnvironment()


# Build the FastAPI app with custom stateful HTTP server
app = FastAPI(
    title="Scheduling RL Environment",
    description="Intelligent Meeting Scheduling Environment for OpenEnv",
    version="1.0.0",
)

_server = SchedulingHTTPEnvServer(
    env=create_scheduling_environment,
    action_cls=SchedulingAction,
    observation_cls=SchedulingObservation,
)
_server.register_routes(app)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
