#!/usr/bin/env python3
"""
Baseline inference script for the Meeting Scheduling RL Environment.

Uses a HEURISTIC policy (BotBooked greedy algorithm) - NO LLM required.
Deterministic, reproducible, fast (~seconds for all 3 tasks).

Output format: [START]/[STEP]/[END] per hackathon spec.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from server.scheduling_env_environment import SchedulingEnvironment
from models import SchedulingAction
from server.scheduling_logic import find_earliest_free_slot, parse_iso


def baseline_policy(obs) -> SchedulingAction:
    """Heuristic baseline using greedy slot search + lowest-priority rescheduling."""

    # Step 1: No proposal yet -> find a free slot
    if obs.current_proposal is None:
        # Build calendars dict from busy_slots
        calendars = {}
        for slot in obs.busy_slots:
            att = slot["attendee"]
            if att not in calendars:
                calendars[att] = []
            calendars[att].append([slot["start"], slot["end"], slot["priority"], slot["summary"]])

        # Try to find a completely free slot
        free = find_earliest_free_slot(
            calendars,
            obs.attendee_ids,
            obs.requested_duration,
            obs.busy_slots[0]["start"] if obs.busy_slots else "2025-04-07T09:00:00+00:00",
            obs.collective_work_hours,
        )

        if free:
            return SchedulingAction(
                action_type="propose_slot",
                proposed_start=free,
                proposed_duration=obs.requested_duration,
            )

        # No completely free slot found.
        # Scan 15-min increments within collective hours for a slot with only
        # reschedulable conflicts (priority > requested_priority).
        min_h = obs.collective_work_hours.get("min_start_hour", 9)
        max_h = obs.collective_work_hours.get("max_end_hour", 17)
        duration = obs.requested_duration
        tz = timezone.utc

        candidate = datetime(2025, 4, 7, min_h, 0, 0, tzinfo=tz)
        end_boundary = datetime(2025, 4, 7, max_h, 0, 0, tzinfo=tz)
        step_delta = timedelta(minutes=15)

        best_candidate = None
        best_conflict_count = 999

        while candidate + timedelta(minutes=duration) <= end_boundary:
            c_start = candidate.isoformat()
            c_end = (candidate + timedelta(minutes=duration)).isoformat()

            # Count conflicts at this candidate
            conflicts_here = []
            for att in obs.attendee_ids:
                for entry in calendars.get(att, []):
                    e_start = parse_iso(entry[0])
                    e_end = parse_iso(entry[1])
                    if candidate < e_end and e_start < candidate + timedelta(minutes=duration):
                        conflicts_here.append(entry)

            # Check if all conflicts are reschedulable
            all_reschedulable = all(
                c[2] > obs.requested_priority for c in conflicts_here
            )

            if all_reschedulable and len(conflicts_here) < best_conflict_count:
                best_candidate = c_start
                best_conflict_count = len(conflicts_here)
                if best_conflict_count == 0:
                    break  # Perfect slot

            candidate += step_delta

        if best_candidate:
            return SchedulingAction(
                action_type="propose_slot",
                proposed_start=best_candidate,
                proposed_duration=duration,
            )

        # Last resort: propose at collective hours start (will likely conflict)
        fallback = f"2025-04-07T{min_h:02d}:00:00+00:00"
        return SchedulingAction(
            action_type="propose_slot",
            proposed_start=fallback,
            proposed_duration=obs.requested_duration,
        )

    # Step 2: Has proposal with conflicts -> reschedule lowest-priority conflict
    if obs.conflicts:
        sorted_conflicts = sorted(obs.conflicts, key=lambda x: x["priority"], reverse=True)
        target = sorted_conflicts[0]

        # Can only reschedule lower priority
        if target["priority"] <= obs.requested_priority:
            return SchedulingAction(action_type="reject")

        # Find a free slot for this attendee to move the meeting to.
        # Search in early morning (06:00-08:00) and late evening (17:00-20:00).
        attendee = target["attendee"]
        meeting_dur = parse_iso(target["end"]) - parse_iso(target["start"])
        dur_min = int(meeting_dur.total_seconds() // 60)

        # Build this attendee's calendar
        att_cal = [
            s for s in obs.busy_slots if s["attendee"] == attendee
        ]
        att_entries = [[s["start"], s["end"], s["priority"], s["summary"]] for s in att_cal]

        new_time = None
        # Try slots at 06:00, 06:30, 07:00, 07:30, 17:00, 17:30, 18:00, 18:30, 19:00
        for h, m in [(6,0),(6,30),(7,0),(7,30),(17,0),(17,30),(18,0),(18,30),(19,0),(19,30),(20,0)]:
            cand = datetime(2025, 4, 7, h, m, 0, tzinfo=timezone.utc)
            cand_end = cand + timedelta(minutes=dur_min)
            cand_iso = cand.isoformat()
            cand_end_iso = cand_end.isoformat()
            # Check free for this attendee
            conflict_found = False
            for e in att_entries:
                es = parse_iso(e[0])
                ee = parse_iso(e[1])
                if cand < ee and es < cand_end:
                    conflict_found = True
                    break
            if not conflict_found:
                new_time = cand_iso
                break

        if not new_time:
            # Give up on this conflict, try rejecting
            return SchedulingAction(action_type="reject")


        return SchedulingAction(
            action_type="reschedule_meeting",
            meeting_id_to_move=target["meeting_id"],
            new_start_time=new_time,
        )

    # Step 3: No conflicts -> finalize
    return SchedulingAction(action_type="finalize")


def main():
    env = SchedulingEnvironment()

    for task_id in ["task1_easy", "task2_medium", "task3_hard"]:
        print(f"[START] task={task_id} env=scheduling_env model=heuristic_baseline")

        obs = env.reset(task_id=task_id)
        done = False
        step = 0
        rewards = []

        while not done and step < 20:
            action = baseline_policy(obs)
            obs = env.step(action)
            done = obs.done
            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)
            step += 1

            error = obs.error_message if obs.error_message else "null"
            print(
                f"[STEP]  step={step} action={action.action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error={error}"
            )

        final_score = rewards[-1] if (done and rewards) else 0.0
        success = obs.success if hasattr(obs, "success") else False
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END]   success={str(success).lower()} steps={step} "
            f"score={final_score:.2f} rewards={rewards_str}"
        )
        print()


if __name__ == "__main__":
    main()
