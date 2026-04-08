"""
inference.py - Meeting Scheduling OpenEnv Agent

Runs an LLM agent through all 3 scheduling tasks and emits structured stdout logs.

Required environment variables:
    API_BASE_URL   LLM API endpoint (OpenAI-compatible)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace / API key

Stdout format (must not deviate):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""
import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# -- Config -------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK    = "scheduling_env"
MAX_STEPS    = 20
TEMPERATURE  = 0.3

TASK_IDS = ["task1_easy", "task2_medium", "task3_hard"]

# -- System prompt ------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an AI meeting scheduling assistant. You must schedule a meeting by choosing actions.

Available actions (respond with EXACTLY one JSON object):

1. Propose a time slot:
   {"action_type": "propose_slot", "proposed_start": "<ISO8601>", "proposed_duration": <minutes>}

2. Reschedule a conflicting meeting (only if priority > requested priority):
   {"action_type": "reschedule_meeting", "meeting_id_to_move": "<attendee>_<start_iso>", "new_start_time": "<ISO8601>"}

3. Finalize the schedule (only when no conflicts remain):
   {"action_type": "finalize"}

4. Reject (give up):
   {"action_type": "reject"}

Rules:
- Propose slots within collective working hours.
- You can only reschedule meetings with LOWER priority (higher number) than the requested meeting.
- meeting_id format is: <attendee>_<start_iso> (e.g., "user1_2025-04-07T09:00:00+00:00").
- After rescheduling all conflicts, call finalize.
- Minimize preference violations and rescheduling.
- Respond with ONLY the JSON object, no other text.
""")


# -- Logging helpers (judge-parsed format) ------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# -- Observation formatting ---------------------------------------------------

def format_observation(obs: Dict[str, Any], step: int) -> str:
    """Convert observation dict into a user prompt for the LLM."""
    max_steps = obs.get("max_steps", MAX_STEPS)
    parts = [
        f"Step {step}/{max_steps}",
        f"Meeting to schedule: {obs.get('requested_duration', '?')} min, priority {obs.get('requested_priority', '?')}",
        f"Attendees: {', '.join(obs.get('attendee_ids', []))}",
    ]

    work_hours = obs.get("collective_work_hours", {})
    parts.append(f"Collective working hours: {work_hours.get('min_start_hour', 9)}:00 - {work_hours.get('max_end_hour', 17)}:00")

    prefs = obs.get("preference_constraints", {})
    if prefs:
        parts.append(
            f"Preferences: max {prefs.get('max_meetings_per_day', 'N/A')} meetings/day, "
            f"buffer required: {prefs.get('requires_buffer', False)}, "
            f"buffer mins: {prefs.get('buffer_minutes', 0)}"
        )

    # Busy slots grouped by attendee
    busy_by_attendee: Dict[str, List] = {}
    for slot in obs.get("busy_slots", []):
        att = slot.get("attendee", "unknown")
        busy_by_attendee.setdefault(att, []).append(slot)

    parts.append("\nCalendars:")
    for att in obs.get("attendee_ids", []):
        slots = busy_by_attendee.get(att, [])
        if slots:
            slot_strs = [
                f"  - {s['start']} to {s['end']} (priority {s['priority']}, {s['summary']})"
                for s in sorted(slots, key=lambda x: x["start"])
            ]
            parts.append(f"  {att}:")
            parts.extend(slot_strs)
        else:
            parts.append(f"  {att}: (no meetings)")

    proposal = obs.get("current_proposal")
    if proposal:
        parts.append(f"\nCurrent proposal: {proposal['start']} to {proposal['end']}")

    conflicts = obs.get("conflicts", [])
    if conflicts:
        parts.append(f"\nConflicts ({len(conflicts)}):")
        for c in conflicts:
            parts.append(
                f"  - {c['attendee']}: {c['start']} to {c['end']} "
                f"(priority {c['priority']}, {c['summary']}, id: {c['meeting_id']})"
            )

    error_msg = obs.get("error_message")
    if error_msg:
        parts.append(f"\nLast error: {error_msg}")

    parts.append(f"\nRescheduled so far: {obs.get('num_rescheduled', 0)}")
    parts.append(f"Preference penalty: {obs.get('preference_penalty', 0.0)}")

    if not proposal and not conflicts:
        parts.append("\nAction needed: propose a time slot for the meeting.")
    elif conflicts:
        parts.append("\nAction needed: reschedule a conflict (lower-priority only) or propose a different slot.")
    else:
        parts.append("\nAction needed: no conflicts remain - you should finalize.")

    return "\n".join(parts)


# -- LLM call -----------------------------------------------------------------

def call_llm(client: OpenAI, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Ask the LLM for the next action given the current observation."""
    user_prompt = format_observation(obs, step)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=512,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(text, obs)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        return fallback_action(obs)


def parse_llm_response(text: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Parse LLM JSON response into an action dict, with fallback."""
    cleaned = text.strip()

    # Handle markdown code blocks
    if "```" in cleaned:
        lines = cleaned.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        cleaned = "\n".join(json_lines).strip()

    # Extract JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        cleaned = cleaned[start:end]

    try:
        data = json.loads(cleaned)
        if "action_type" not in data:
            raise ValueError("No action_type in response")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[DEBUG] Parse error: {e}. Response: {text[:200]}", file=sys.stderr, flush=True)
        return fallback_action(obs)


def fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a safe fallback action based on current observation state."""
    if obs.get("current_proposal") is None:
        min_h = obs.get("collective_work_hours", {}).get("min_start_hour", 9)
        duration = obs.get("requested_duration", 30)
        return {
            "action_type": "propose_slot",
            "proposed_start": f"2025-04-07T{min_h:02d}:00:00+00:00",
            "proposed_duration": duration,
        }
    elif not obs.get("conflicts"):
        return {"action_type": "finalize"}
    else:
        return {"action_type": "reject"}


# -- Episode runner -----------------------------------------------------------

def run_episode(client: OpenAI, task_id: str) -> None:
    """Run one full episode for a task, emitting [START]/[STEP]/[END] logs."""
    import requests

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        try:
            resp = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": task_id},
                timeout=30,
            )
            resp.raise_for_status()
            reset_data = resp.json()
        except Exception as e:
            print(f"[DEBUG] Reset failed: {e}", file=sys.stderr, flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return

        observation = reset_data.get("observation", reset_data)
        done = reset_data.get("done", False)

        # Episode loop
        while not done and steps_taken < MAX_STEPS:
            steps_taken += 1

            # Get action from LLM
            action = call_llm(client, observation, steps_taken)
            action_type = action.get("action_type", "unknown")

            # Build compact action string for logging
            if action_type == "propose_slot":
                action_str = f"propose_slot({action.get('proposed_start', '?')[:16]},{action.get('proposed_duration', '?')}m)"
            elif action_type == "reschedule_meeting":
                action_str = f"reschedule({action.get('meeting_id_to_move', '?')[:20]})"
            else:
                action_str = action_type

            # Execute step
            try:
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": action},
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()
            except Exception as e:
                print(f"[DEBUG] Step failed: {e}", file=sys.stderr, flush=True)
                rewards.append(0.0)
                log_step(step=steps_taken, action=action_str, reward=0.0, done=True, error=str(e))
                break

            observation = step_data.get("observation", {})
            reward = step_data.get("reward", 0.0) or 0.0
            done = step_data.get("done", False)
            error = observation.get("error_message")

            rewards.append(reward)
            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error)

        # Final score is the last reward (0.0-1.0 from calculate_final_reward)
        score = rewards[-1] if rewards else 0.0
        # Clamp to (0.01, 0.99) as required by judge
        score = max(0.01, min(score, 0.99))
        success = score > 0.3

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# -- Main ---------------------------------------------------------------------

def main():
    global ENV_URL

    parser = argparse.ArgumentParser(description="Scheduling env baseline inference")
    parser.add_argument("--task", choices=TASK_IDS, help="Run a specific task only")
    parser.add_argument("--all", action="store_true", help="Run all 3 tasks (default)")
    parser.add_argument("--url", default=ENV_URL, help="Environment base URL")
    args = parser.parse_args()

    ENV_URL = args.url

    # Check for TASK_NAME environment variable (judge may set this)
    target_task = os.getenv("TASK_NAME")
    if target_task:
        if "task1" in target_task or "easy" in target_task:
            args.task = "task1_easy"
        elif "task2" in target_task or "medium" in target_task:
            args.task = "task2_medium"
        elif "task3" in target_task or "hard" in target_task:
            args.task = "task3_hard"

    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks = [args.task] if args.task else TASK_IDS

    for task_id in tasks:
        run_episode(client, task_id)


if __name__ == "__main__":
    main()
