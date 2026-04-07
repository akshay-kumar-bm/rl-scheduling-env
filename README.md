---
title: Scheduling Env Environment Server
emoji: 📅
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Meeting Scheduling RL Environment

An OpenEnv reinforcement-learning environment where AI agents learn to schedule meetings optimally across multiple attendees. The agent must propose time slots, resolve calendar conflicts by rescheduling lower-priority meetings, and satisfy each participant's scheduling preferences — all within a limited number of steps.

## Overview

The environment simulates a realistic corporate scheduling assistant. Given a meeting request, the agent iteratively:

1. **Proposes** a time slot for all required attendees.
2. **Reschedules** any lower-priority conflicting meetings to free up the slot.
3. **Finalizes** the booking once the slot is conflict-free.

Each episode is scored on scheduling quality (0.0–1.0), penalizing preference violations, unnecessary rescheduling, and excessive steps.

## Quick Start

### Running the Heuristic Baseline (no LLM needed)

```bash
python inference.py
```

This runs a greedy baseline policy across all three tasks and prints step-by-step output in the required `[START]`/`[STEP]`/`[END]` format.

### Using the Environment Directly (Python)

```python
from server.scheduling_env_environment import SchedulingEnvironment
from models import SchedulingAction

env = SchedulingEnvironment()

# Reset to a specific task
obs = env.reset(task_id="task1_easy")
print(f"Attendees: {obs.attendee_ids}")
print(f"Duration:  {obs.requested_duration} min")
print(f"Priority:  {obs.requested_priority}")

# Propose a time slot
result = env.step(SchedulingAction(
    action_type="propose_slot",
    proposed_start="2025-04-07T10:00:00+00:00",
    proposed_duration=30,
))
print(f"Conflicts: {result.conflicts}")
print(f"Reward:    {result.reward}")

# Finalize when conflict-free
result = env.step(SchedulingAction(action_type="finalize"))
print(f"Success: {result.success}  Final score: {result.reward:.2f}")
```

### Using the HTTP Client

```python
from client import SchedulingEnv
from models import SchedulingAction

with SchedulingEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_id="task2_medium")
    obs = result.observation

    # Propose a slot
    result = env.step(SchedulingAction(
        action_type="propose_slot",
        proposed_start="2025-04-07T11:00:00+00:00",
        proposed_duration=60,
    ))

    # Reschedule a conflicting lower-priority meeting
    if result.observation.conflicts:
        conflict = result.observation.conflicts[0]
        result = env.step(SchedulingAction(
            action_type="reschedule_meeting",
            meeting_id_to_move=conflict["meeting_id"],
            new_start_time="2025-04-07T07:00:00+00:00",
        ))

    # Finalize
    result = env.step(SchedulingAction(action_type="finalize"))
    print(f"Score: {result.reward:.2f}")
```

## Environment Details

### Actions (`SchedulingAction`)

| `action_type`        | Required fields                              | Description                                               |
|----------------------|----------------------------------------------|-----------------------------------------------------------|
| `propose_slot`       | `proposed_start`, `proposed_duration`        | Propose a meeting start time (ISO 8601) and duration (min)|
| `reschedule_meeting` | `meeting_id_to_move`, `new_start_time`       | Move a lower-priority conflict to a new time              |
| `finalize`           | _(none)_                                     | Confirm the proposed slot; ends the episode               |
| `reject`             | _(none)_                                     | Give up on scheduling; ends the episode with 0 reward     |

**Meeting ID format:** `{attendee}_{start_iso}` — e.g. `user1_2025-04-07T09:00:00+00:00`

### Observations (`SchedulingObservation`)

| Field                   | Type                    | Description                                                  |
|-------------------------|-------------------------|--------------------------------------------------------------|
| `requested_duration`    | `int`                   | Meeting duration in minutes                                  |
| `requested_priority`    | `int`                   | Priority of the new meeting (1 = highest, 5 = lowest)        |
| `attendee_ids`          | `List[str]`             | Required attendees                                           |
| `busy_slots`            | `List[dict]`            | All existing calendar entries for attendees                  |
| `collective_work_hours` | `dict`                  | Shared working-hours window `{min_start_hour, max_end_hour}` |
| `preference_constraints`| `dict`                  | Aggregated constraints (max meetings/day, buffer, etc.)      |
| `current_proposal`      | `dict \| None`          | Currently proposed slot `{start, end}`                       |
| `conflicts`             | `List[dict]`            | Conflicts for the current proposal                           |
| `preference_penalty`    | `float`                 | Accumulated preference-violation penalty                     |
| `num_rescheduled`       | `int`                   | Meetings rescheduled so far in this episode                  |
| `steps_taken`           | `int`                   | Steps used so far                                            |
| `max_steps`             | `int`                   | Episode step limit (20)                                      |
| `success`               | `bool`                  | `True` when the meeting is successfully booked               |
| `error_message`         | `str \| None`           | Reason if the last action was invalid                        |
| `done`                  | `bool`                  | `True` when the episode has ended                            |
| `reward`                | `float`                 | Step or final reward                                         |

### Reward Design

**Step-level rewards** (returned after each `propose_slot` or `reschedule_meeting`):

| Outcome                                  | Reward |
|------------------------------------------|--------|
| Conflict-free proposal (low penalty)     | +0.5   |
| Proposal has reschedulable conflicts     | +0.2   |
| Proposal has non-reschedulable conflicts | −0.3   |
| Invalid action                           | −0.1   |
| Outside working hours                    | −0.2   |

**Final reward** (returned on `finalize`) — deducted from 1.0:

```
preference_deduction  = min(0.75, (penalty ** 1.2) / 200.0)
reschedule_deduction  = min(0.30, 0.05 * (1.8 ** num_rescheduled))   [if any rescheduled]
time_deduction        = steps_taken * 0.015

final_reward = clamp(1.0 - preference_deduction - reschedule_deduction - time_deduction, 0.0, 1.0)
```

Timeout (step 20 reached without `finalize`) gives partial credit: 70 % of the theoretical reward if conflict-free, or a progress-based fraction otherwise.

## Tasks

Three tasks of increasing difficulty are provided as JSON scenarios in `server/scenarios/`:

| Task ID         | Difficulty | Attendees | Duration | Priority | Rescheduling needed | Expected score |
|-----------------|------------|-----------|----------|----------|---------------------|----------------|
| `task1_easy`    | Easy       | 2         | 30 min   | 3        | No                  | 0.8 – 1.0      |
| `task2_medium`  | Medium     | 4         | 60 min   | 2        | Yes (1 meeting)     | 0.5 – 0.7      |
| `task3_hard`    | Hard       | 6         | 45 min   | 2        | Yes (3+ meetings)   | 0.25 – 0.45    |

### task1_easy — Team Sync (2 attendees)

- Two attendees each have 2 existing meetings; a clear free slot exists at **10:00**.
- Agent should find the free slot and finalize in 2 steps.
- No rescheduling required.

### task2_medium — Cross-Team Planning (4 attendees)

- Four attendees with densely packed schedules; the optimal slot at **11:00** has one low-priority conflict (`user3` Coffee chat, priority 4).
- Agent needs to propose the slot, reschedule the conflict, then finalize.
- User preferences include back-to-back avoidance and different preferred-hour windows.

### task3_hard — Executive Planning Session (6 attendees)

- Six attendees with very dense calendars; the best window at **15:00** requires rescheduling three low-priority meetings (priority 4).
- Multiple valid solutions exist; the agent must navigate cascading constraints.
- All attendees have strict buffer requirements and narrow preferred-hour windows.

## Participant Preferences

Each attendee can have the following preferences (stored in scenario JSON and observed via `preference_constraints`):

| Preference             | Description                                         | Penalty for violation |
|------------------------|-----------------------------------------------------|-----------------------|
| `preferred_hours`      | `{start: H, end: H}` — preferred working hours      | +50 per participant   |
| `max_meetings_per_day` | Maximum meetings the participant wants in a day      | +30 per participant   |
| `avoid_back_to_back`   | Whether a buffer gap is required between meetings    | +20 per participant   |
| `buffer_minutes`       | Gap required before/after a meeting (if avoid_btb)  | (part of above)       |

The **collective working hours** (the intersection of all attendees' preferred hours) define the hard constraint window within which proposals must fall.

## API Endpoints

The server exposes the following HTTP endpoints (also available via the Web UI at `/web`):

| Method | Path      | Description                                                        |
|--------|-----------|--------------------------------------------------------------------|
| POST   | `/reset`  | Start a new episode. Body: `{"task_id": "task1_easy"}`             |
| POST   | `/step`   | Take an action. Body: `{"action_type": "...", ...action fields}`   |
| GET    | `/state`  | Return the full internal `SchedulingState`                         |
| GET    | `/health` | Health check — returns `{"status": "healthy"}`                     |
| GET    | `/docs`   | Interactive OpenAPI / Swagger UI                                   |

### Example: REST interaction

```bash
# Start episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_easy"}'

# Propose a slot
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "propose_slot", "proposed_start": "2025-04-07T10:00:00+00:00", "proposed_duration": 30}'

# Finalize
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "finalize"}'
```

## Development & Testing

### Run the baseline inference script

```bash
python inference.py
```

### Start the server locally

```bash
uvicorn server.app:app --reload
```

### Validate the environment (required before submission)

```bash
openenv validate
```

### Generate / update the lock file

```bash
uv lock
```

### Build the Docker image

```bash
docker build -t scheduling_env:latest .
```

## Deploying to Hugging Face Spaces

```bash
# From the project root (where openenv.yaml is located)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-scheduling-env

# Push as a private space
openenv push --private
```

The `openenv push` command validates the environment, builds a Hugging Face-compatible Docker image, and uploads it. After deployment your space is available at:

```
https://huggingface.co/spaces/<repo-id>
```

The deployed space includes:
- **Web Interface** at `/web` — interactive UI for exploring the environment
- **API Documentation** at `/docs` — full OpenAPI / Swagger interface
- **Health Check** at `/health` — container health monitoring

### Options

| Flag | Description |
|------|-------------|
| `--directory`, `-d` | Directory with `openenv.yaml` (default: current dir) |
| `--repo-id`, `-r` | Repository ID `username/repo-name` |
| `--base-image`, `-b` | Override Dockerfile `FROM` image |
| `--private` | Deploy as a private space (default: public) |

## Environment Variables (for LLM-based inference)

Create a `.env` file (never commit it):

```
API_BASE_URL=https://router.huggingface.co/v1   # HF Router endpoint
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct            # Model identifier
HF_TOKEN=hf_...                                  # Hugging Face API key
```

## Project Structure

```
rl-scheduling-env/
├── Dockerfile                          # Container image (root, required by openenv)
├── README.md                           # This file
├── openenv.yaml                        # OpenEnv manifest
├── pyproject.toml                      # Project metadata and dependencies
├── uv.lock                             # Locked dependencies (generated by `uv lock`)
├── __init__.py                         # Package exports
├── models.py                           # Pydantic models: SchedulingAction,
│                                       #   SchedulingObservation, SchedulingState
├── client.py                           # SchedulingEnv HTTP/WebSocket client
├── inference.py                        # Heuristic baseline (no LLM required)
└── server/
    ├── __init__.py                     # Server package exports
    ├── app.py                          # FastAPI app + SchedulingHTTPEnvServer
    ├── scheduling_env_environment.py   # Core RL environment (reset / step / state)
    ├── scheduling_logic.py             # Pure utility functions (conflict detection,
    │                                   #   preference scoring, reward calculation)
    ├── graders.py                      # SchedulingGrader (0.0–1.0 episode scorer)
    ├── requirements.txt                # Server-side Python dependencies
    └── scenarios/
        ├── task1_easy.json             # Easy: 2 attendees, free slot exists
        ├── task2_medium.json           # Medium: 4 attendees, 1 rescheduling needed
        └── task3_hard.json             # Hard: 6 attendees, 3+ reschedulings needed
```
