# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

OpenEnv RL environment for the **Meta OpenEnv Hackathon**. Implements an intelligent meeting scheduling environment where AI agents learn to schedule meetings across multiple attendees by proposing time slots, rescheduling lower-priority conflicts, and balancing participant preferences.

## Development Commands

```bash
# Run baseline inference (heuristic, no LLM needed)
python inference.py

# Start server locally
uvicorn server.app:app --reload

# Validate environment for submission
openenv validate

# Generate/update lock file (required by validator)
uv lock

# Deploy to Hugging Face Spaces
openenv push

# Build Docker image (Dockerfile must be in root)
docker build -t scheduling_env:latest .
```

## Architecture

### OpenEnv Interface (client-server pattern)

The environment follows OpenEnv's standard API:
- **`POST /reset`** — starts a new episode, accepts `{"task_id": "task1_easy"}`. Returns observation.
- **`POST /step`** — takes an action, returns observation with reward/done.
- **`GET /state`** — returns internal environment state.
- **`GET /health`** — health check.

### Core Flow

`server/app.py` creates a `SchedulingHTTPEnvServer` (subclasses `HTTPEnvServer`) that wraps a persistent `SchedulingEnvironment` instance. The server registers custom `/reset`, `/step`, `/state` routes.

`server/scheduling_env_environment.py` — Main environment class implementing `Environment`. Loads JSON scenarios from `server/scenarios/`, processes 4 action types: `propose_slot`, `reschedule_meeting`, `finalize`, `reject`. Episode ends on `finalize`, `reject`, or timeout (20 steps).

`server/scheduling_logic.py` — Pure utility functions: conflict detection, preference scoring, reward calculation, free-slot search. All datetime handling uses timezone-aware ISO 8601 strings. Calendar format: `Dict[str, List[List]]` where each entry is `[start_iso, end_iso, priority_int, summary_str]`.

`models.py` — Pydantic models (`SchedulingAction`, `SchedulingObservation`, `SchedulingState`) imported by both server and client.

`client.py` — `SchedulingEnv` extends `EnvClient` for WebSocket-based interaction.

`inference.py` — Heuristic baseline (no LLM). Greedy free-slot search + lowest-priority rescheduling. Must emit `[START]`/`[STEP]`/`[END]` stdout format.

### Reward Design

Reward is multi-component, deducted from 1.0 (see `calculate_final_reward` in `scheduling_logic.py`):
- Preference penalty: violations of preferred hours (+50), max meetings/day (+30), back-to-back (+20)
- Rescheduling deduction: exponential penalty per meeting moved
- Time deduction: 0.015 per step taken

Step-level rewards: +0.5 (conflict-free proposal), +0.2 (reschedulable conflicts), -0.3 (non-reschedulable conflicts), -0.1/-0.2 (invalid actions).

### Tasks (3 difficulty levels)

JSON scenarios in `server/scenarios/`:
- **task1_easy** — 2 attendees, free slot exists, no rescheduling needed. Expected score: 0.8–1.0
- **task2_medium** — 3 attendees, requires 1 rescheduling. Expected score: 0.5–0.8
- **task3_hard** — 4 attendees, multiple overlapping conflicts, cascading rescheduling. Expected score: 0.2–0.6

### Key Constraint: Meeting IDs

Format is `{attendee}_{start_iso}` (e.g., `user1_2025-04-07T09:00:00+00:00`). Used by `_find_meeting()` to look up calendar entries for rescheduling.

## Hackathon Submission Requirements

- `openenv validate` must pass
- Dockerfile in root directory (not `/server`)
- `inference.py` in root, uses `[START]`/`[STEP]`/`[END]` stdout format
- 3+ tasks with graders scoring 0.0–1.0 with diverse scores
- Runtime < 20 minutes on vcpu=2, memory=8GB
- Deploy via `openenv push` to HF Spaces

## Environment Variables (for LLM-based inference)

Defined in `.env` (never commit):
```
API_BASE_URL    # HF Router endpoint (default: https://router.huggingface.co/v1)
MODEL_NAME      # Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
HF_TOKEN        # Hugging Face API key
```
