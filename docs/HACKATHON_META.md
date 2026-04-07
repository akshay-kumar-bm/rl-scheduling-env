# Meta OpenEnv Hackathon - Round 1

## Overview

Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

## Task Requirements

### Must-Have Features

1. **Real-world Task Simulation**
   - Must simulate tasks humans actually do
   - Not games or toys
   - Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation

2. **OpenEnv Spec Compliance**
   - Typed Observation, Action, and Reward Pydantic models
   - `step(action)` → returns observation, reward, done, info
   - `reset()` → returns initial observation
   - `state()` → returns current state
   - `openenv.yaml` with metadata
   - Must pass `openenv validate`

3. **Minimum 3 Tasks with Agent Graders**
   - Each task defines a concrete objective
   - Programmatic grader scoring (0.0–1.0)
   - Difficulty range: easy → medium → hard
   - Clear, deterministic success/failure criteria

4. **Meaningful Reward Function**
   - Provides signal over full trajectory (not just binary)
   - Rewards partial progress toward completion
   - Penalizes undesirable behavior (infinite loops, destructive actions)

5. **Baseline Inference Script**
   - Uses OpenAI API client
   - Reads credentials from `OPENAI_API_KEY` environment variable
   - Produces reproducible baseline scores on all 3 tasks

## Non-Functional Requirements

### Deployment
- **Hugging Face Space**: Environment must run as containerized HF Space tagged with `openenv`
- **Dockerfile**: Working containerization with clean `docker build + docker run`

### Documentation
README must include:
- Environment description and motivation
- Action and observation space definitions
- Task descriptions with expected difficulty
- Setup and usage instructions
- Baseline scores

## Evaluation Criteria & Scoring

### Scoring Breakdown (100 points)

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Real-world utility** | 30% | Does the environment model a genuine task? Would someone use this for training/evaluating agents? |
| **Task & grader quality** | 25% | Well-defined tasks with clear objectives? Accurate graders? Meaningful difficulty progression? |
| **Environment design** | 20% | Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries |
| **Code quality & spec compliance** | 15% | Follows OpenEnv spec, clean structure, typed models, documented, tested, working Dockerfile |
| **Creativity & novelty** | 10% | Novel problem domain, interesting mechanics, clever reward design, original approach |

### Detailed Scoring Rubrics

#### Real-world Utility (30%)
- **0–5**: Toy/artificial problem with no practical application
- **6–15**: Valid domain but shallow modeling
- **16–25**: Good domain modeling, useful for agent evaluation
- **26–30**: Excellent — fills real gap, immediate value for RL/agent community

#### Task & Grader Quality (25%)
- 3+ tasks with difficulty range?
- Graders produce scores between 0.0–1.0?
- Graders deterministic and reproducible?
- Hard task genuinely challenges frontier models?

#### Environment Design (20%)
- `reset()` produces clean state?
- Action/observation types well-designed and documented?
- Reward function provides useful varying signal (not sparse)?
- Episode boundaries sensible?

#### Code Quality & Spec Compliance (15%)
- `openenv validate` passes?
- `docker build && docker run` works?
- HF Space deploys and responds?
- Baseline script runs and reproduces scores?

#### Creativity & Novelty (10%)
- Domain not seen in OpenEnv before?
- Reward design has interesting properties?
- Clever mechanics that make environment engaging?

## Judging Process

### Phase 1: Automated Validation (Pass/Fail Gate)
- HF Space deploys
- OpenEnv spec compliance
- Dockerfile builds
- Baseline reproduces
- 3+ tasks with graders

### Phase 2: Agentic Evaluation (Scored)
- Baseline agent re-run
- Standard Open LLM agent (e.g., Nemotron 3 Super) run against all environments
- Score variance check

### Phase 3: Human Review
Top submissions reviewed by Meta and Hugging Face engineers for:
- Real-world utility
- Creativity
- Exploit checks

### Disqualification Criteria
- Environment does not deploy or respond
- Plagiarized or trivially modified existing environments
- Graders that always return the same score
- No baseline inference script

## Pre-Submission Checklist

All must pass or you're disqualified:

- [ ] HF Space deploys (200 response to reset())
- [ ] OpenEnv spec compliance validated
- [ ] Dockerfile builds successfully
- [ ] Baseline script reproduces without error
- [ ] 3+ tasks with graders (scores in 0.0–1.0 range)

## Mandatory Requirements

### Environment Variables
Must be defined in your environment configuration:

```bash
API_BASE_URL    # The API endpoint for the LLM
MODEL_NAME      # The model identifier to use for inference
HF_TOKEN        # Your Hugging Face / API key
LOCAL_IMAGE_NAME # (Optional) Name of local image if using from_docker_image()
```

### Script Requirements
- **Filename**: `inference.py` (must be in root directory)
- **LLM Calls**: Must use OpenAI Client with above variables
- **Logging Format**: Must follow [START], [STEP], [END] format (see below)

### Infrastructure Restrictions
- **Runtime**: Inference script must complete in < 20 minutes
- **Resources**: Must run on vcpu=2, memory=8GB

## STDOUT Logging Format

### Required Format
The script must emit exactly three line types to stdout, in this order:

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

### Format Rules
- One [START] line at episode begin
- One [STEP] line per step, immediately after `env.step()` returns
- One [END] line after `env.close()`, always emitted (even on exception)
- `reward` and `rewards` formatted to 2 decimal places
- `done` and `success` are lowercase booleans: `true` or `false`
- `error` is the raw `last_action_error` string, or `null` if none
- All fields on a single line with no newlines within a line
- Each task should return score in [0, 1]

### Example Output
```
[START] task=click-test env=miniwob model=Qwen3-VL-30B
[STEP] step=1 action=click('123') reward=0.00 done=false error=null
[STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
[STEP] step=3 action=click('789') reward=1.00 done=true error=null
[END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
```

## Sample Inference Script

```python
"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env

IMAGE_NAME = os.getenv("IMAGE_NAME")  # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.7

# TODO: Implement the rest of your inference script here
```

## Pre-Validation Script

```bash
#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   Or download and run locally:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BOLD=''
  NC=''
fi

# TODO: Add the rest of the validation script
```

## Tips for Success

1. **Choose a Real Problem**: Pick a task that has genuine value for the AI/agent community
2. **Design Good Rewards**: Provide meaningful signals throughout the episode, not just at the end
3. **Test Thoroughly**: Ensure your environment works cleanly with `docker build && docker run`
4. **Document Well**: Clear README helps reviewers understand your contribution
5. **Start Simple**: Get the basic OpenEnv spec working first, then add complexity
6. **Run Validator**: Use the pre-validation script before submitting

## Resources

- OpenEnv Documentation: [Link to be added]
- Hugging Face Spaces: https://huggingface.co/spaces
- OpenAI API Client: https://platform.openai.com/docs/api-reference

## Submission Deadline

[To be announced]

---

**Good luck with your submission! 🚀**
