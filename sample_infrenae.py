"""
LLM-based Inference Script for Meeting Scheduling RL Environment.
===================================
Uses OpenAI-compatible LLM via HF Router to intelligently schedule meetings.

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=scheduling_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from scheduling_env.client import SchedulingEnv
from scheduling_env.models import SchedulingAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

ENV_REPO_ID = "Akshaykumarbm/scheduling_env"
BENCHMARK = "scheduling_env"
TASKS = ["task1_easy", "task2_medium", "task3_hard"]
MAX_STEPS = 20
TEMPERATURE = 0.3
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

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


def format_observation(obs, step: int) -> str:
    """Convert a SchedulingObservation into a user prompt for the LLM."""
    parts = [
        f"Step {step}/{obs.max_steps}",
        f"Meeting to schedule: {obs.requested_duration} min, priority {obs.requested_priority}",
        f"Attendees: {', '.join(obs.attendee_ids)}",
        f"Collective working hours: {obs.collective_work_hours.get('min_start_hour', 9)}:00 - {obs.collective_work_hours.get('max_end_hour', 17)}:00",
    ]

    if obs.preference_constraints:
        parts.append(f"Preferences: max {obs.preference_constraints.get('max_meetings_per_day', 'N/A')} meetings/day, "
                      f"buffer required: {obs.preference_constraints.get('requires_buffer', False)}, "
                      f"buffer mins: {obs.preference_constraints.get('buffer_minutes', 0)}")

    # Busy slots grouped by attendee
    busy_by_attendee: Dict[str, List] = {}
    for slot in obs.busy_slots:
        att = slot["attendee"]
        busy_by_attendee.setdefault(att, []).append(slot)

    parts.append("\nCalendars:")
    for att in obs.attendee_ids:
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

    if obs.current_proposal:
        parts.append(f"\nCurrent proposal: {obs.current_proposal['start']} to {obs.current_proposal['end']}")

    if obs.conflicts:
        parts.append(f"\nConflicts ({len(obs.conflicts)}):")
        for c in obs.conflicts:
            parts.append(
                f"  - {c['attendee']}: {c['start']} to {c['end']} "
                f"(priority {c['priority']}, {c['summary']}, id: {c['meeting_id']})"
            )

    if obs.error_message:
        parts.append(f"\nLast error: {obs.error_message}")

    parts.append(f"\nRescheduled so far: {obs.num_rescheduled}")
    parts.append(f"Preference penalty: {obs.preference_penalty}")

    if not obs.current_proposal and not obs.conflicts:
        parts.append("\nAction needed: propose a time slot for the meeting.")
    elif obs.conflicts:
        parts.append("\nAction needed: reschedule a conflict (lower-priority only) or propose a different slot.")
    else:
        parts.append("\nAction needed: no conflicts remain - you should finalize.")

    return "\n".join(parts)


def parse_llm_response(text: str, obs) -> SchedulingAction:
    """Parse LLM JSON response into a SchedulingAction, with fallback."""
    # Extract JSON from response (handle markdown code blocks)
    cleaned = text.strip()
    if "```" in cleaned:
        # Extract content between code fences
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

    # Try to find JSON object in the response
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        cleaned = cleaned[start:end]

    try:
        data = json.loads(cleaned)
        return SchedulingAction(**data)
    except (json.JSONDecodeError, Exception) as e:
        print(f"[DEBUG] Failed to parse LLM response: {e}. Response: {text[:200]}", flush=True)
        # Fallback: if we have no proposal yet, propose at first available hour
        if obs.current_proposal is None:
            min_h = obs.collective_work_hours.get("min_start_hour", 9)
            return SchedulingAction(
                action_type="propose_slot",
                proposed_start=f"2025-04-07T{min_h:02d}:00:00+00:00",
                proposed_duration=obs.requested_duration,
            )
        elif not obs.conflicts:
            return SchedulingAction(action_type="finalize")
        else:
            return SchedulingAction(action_type="reject")


def get_llm_action(client: OpenAI, obs, step: int) -> SchedulingAction:
    """Query the LLM and return a SchedulingAction."""
    user_prompt = format_observation(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(text, obs)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return parse_llm_response("", obs)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_task(env, client: OpenAI, task_id: str) -> None:
    """Run a single scheduling task."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_llm_action(client, obs, step)

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.error_message

            rewards.append(reward)
            steps_taken = step

            action_str = action.action_type
            if action.action_type == "propose_slot":
                action_str = f"propose_slot({action.proposed_start},{action.proposed_duration}m)"
            elif action.action_type == "reschedule_meeting":
                action_str = f"reschedule({action.meeting_id_to_move}->{action.new_start_time})"

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Score is the final reward (0.0-1.0 from calculate_final_reward)
        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = obs.success if hasattr(obs, "success") else (score > 0.0)

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await SchedulingEnv.from_env(ENV_REPO_ID)

    try:
        for task_id in TASKS:
            await run_task(env, llm_client, task_id)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
