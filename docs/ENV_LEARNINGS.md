# OpenEnv Environment Research - Key Learnings

Research conducted on 5 top OpenEnv environments to inform hackathon project development.

## Executive Summary

| Environment | Domain | Key Strength | Best For Learning |
|-------------|--------|--------------|-------------------|
| **calendar_env** | Calendar Management | Generic MCP wrapper architecture | Multi-tenant systems, database-backed tasks |
| **reasoning_gym_env** | Reasoning Tasks | Minimal, single-step episodes | Simple task structures, dataset integration |
| **tbench2_env** | Terminal/Tool Use | Dual execution modes (local/docker) | Tool benchmarking, session management |
| **carla_env** | Autonomous Driving | Scenario-based design | Complex simulations, ethical dilemmas |
| **repl_env** | Code Execution | Recursive LLM architecture | Interactive environments, reward shaping |

---

## 1. Calendar Environment (calendar_env)

### Architecture Highlights
- **Generic MCP Wrapper**: Fully reusable `openenv_wrapper/` for any MCP server
- **Multi-Tenancy**: SQLite per agent via `x-database-id` header
- **Rich Database Schema**: Google Calendar API v3 compliant models

### Action/Observation Pattern
```python
# Action
class MCPAction(Action):
    action_type: Literal["ListToolsAction", "ToolCallAction"]
    tool_name: Optional[str]
    arguments: Optional[Dict]

# Observation
class MCPObservation(Observation):
    success: bool
    error_message: Optional[str]
    tool_result: Optional[Dict]
    reward: Optional[float]
    done: bool
```

### Task Definition Pattern
- **JSON Scenarios**: Version-controlled task definitions
- **SQL Verifiers**: Programmatic graders checking database state
- **3 Verifier Types**: database_state, response_check, tool_execution

### Reward Design
- Sparse binary rewards: +1.0 (success), -0.5 (error)
- ListToolsAction: +0.1 (discovery reward)
- Status code based with metadata for flexibility

### Worth Copying
1. **Generic wrapper architecture** - Copy `openenv_wrapper/` for new MCPs
2. **Session manager pattern** - Multi-tenant database isolation
3. **Verifier-driven tasks** - No code changes for new tasks
4. **Config-driven tool discovery** - Dynamic tool handlers via importlib

---

## 2. Reasoning Gym Environment (reasoning_gym_env)

### Architecture Highlights
- **Minimal footprint**: ~200 lines core logic
- **Single-step episodes**: reset() → step() → done
- **Dataset persistence**: Reuse datasets across resets

### Action/Observation Pattern
```python
# Action
class ReasoningGymAction(Action):
    answer: str  # Agent's answer

# Observation
class ReasoningGymObservation(Observation):
    question: Optional[str]      # Only in reset()
    score: Optional[float]       # Only after step()
    correct_answer: Optional[str]
    done: bool
```

### Task Definition Pattern
- **External library**: `reasoning_gym` handles generation + scoring
- **Simple datasets**: Single task type (leg_counting, reverse_sort, etc.)
- **Composite datasets**: Mix multiple tasks with weights

### Reward Design
- **Binary/partial**: Depends on dataset scoring function
- **Terminal only**: reward=0.0 on reset, actual score after step()
- **Single-step**: No trajectory rewards

### Worth Copying
1. **Iterator pattern** - Seamless dataset cycling with StopIteration handling
2. **Parameter idempotency** - reset() continues, reset(seed=...) restarts
3. **Dataset caching** - Compare config to avoid rebuilding
4. **Minimal state** - Just episode_id and step_count

---

## 3. TB2 Environment (tbench2_env)

### Architecture Highlights
- **Dual execution modes**: Local (CAMEL toolkit) vs Docker (TB2 fidelity)
- **Session management**: Streaming process support via session_id
- **Task auto-discovery**: Download from GitHub + cache locally

### Action/Observation Pattern
```python
# Action
class Tbench2Action(Action):
    action_type: str  # exec, write, view, wait, kill, evaluate, etc.
    command: str
    session_id: Optional[str]
    block: bool = True

# Observation
class Tbench2Observation(Observation):
    instruction: str
    output: str
    success: bool
    error: str
    reward: Optional[float]  # Only on evaluate
    done: bool              # Only on evaluate
```

### Task Definition Pattern
- **TOML-based**: `task.toml` with environment + verifier config
- **Pytest graders**: Each task has tests/ directory
- **External benchmark**: Terminal-Bench 2 suite

### Reward Design
- **Binary**: 1.0 if all pytest tests pass, 0.0 otherwise
- **Terminal only**: reward=None until evaluate action
- **Exit code parsing**: `__TB2_EXIT_CODE__:$?` marker pattern

### Worth Copying
1. **Dual mode pattern** - Local + Docker execution with env var switching
2. **Lazy dependency loading** - Import errors surface only when used
3. **Docker-in-Docker safe** - Tar streaming instead of bind mounts
4. **Session isolation** - Unique working directories per episode_id
5. **Metadata-driven discovery** - Tasks self-describe requirements

---

## 4. CARLA Environment (carla_env)

### Architecture Highlights
- **Scenario system**: BaseScenario ABC with composable tasks
- **Rubric factory**: Auto-select reward function by scenario type
- **Mock mode**: Test without GPU/CARLA
- **GPU-accelerated**: T4 16GB minimum for real mode

### Action/Observation Pattern
```python
# Action
class CarlaAction(Action):
    action_type: str  # observe, control, navigate, capture_image, etc.
    throttle: Optional[float]  # [0, 1] with Pydantic validation
    steer: Optional[float]     # [-1, 1]
    brake: Optional[float]     # [0, 1]

# Observation
class CarlaObservation(Observation):
    scene_description: str
    vehicle_state: Dict  # speed, location, rotation
    collision_detected: bool
    nearby_actors: List[Dict]
    camera_images: Optional[Dict]
    rubric_reward: float
```

### Task Definition Pattern
- **9 Trolley scenarios**: Ethical dilemmas with expected outcomes
- **Navigation tasks**: Maze (goal-directed), Free-roam (open-world)
- **JSON externalized**: Benchmark definitions separate from code

### Reward Design
- **Trajectory-based (Trolley)**: r_t = 0.0 until terminal, then gamma-discounted final
- **Step-level (Navigation)**: Progress + arrival bonus - collision penalty - time cost
- **Scenario-specific**: compute_outcome() owns scoring logic

### Worth Copying
1. **Scenario ABC** - Each task owns physics + scoring independently
2. **Rubric factory** - Auto-select reward function by task type
3. **Dual mode** - Mock for testing, real for evaluation
4. **Layered config** - Common + scenario-specific fields
5. **JSON externalization** - Decouple task data from code

---

## 5. REPL Environment (repl_env)

### Architecture Highlights
- **Layered design**: Environment → Runner → Backend separation
- **Recursive LLM**: Depth-limited child spawning with RLM pattern
- **Composable rubrics**: Outcome + process rewards
- **Thread-safe batching**: Multiple concurrent child queries

### Action/Observation Pattern
```python
# Action
class REPLAction(Action):
    code: str
    is_final: bool = False
    final_answer: Optional[str] = None

# Observation
class REPLObservation(Observation):
    result: CodeBlockResult  # stdout, stderr, locals_snapshot
    available_variables: List[str]
    iteration: int
    done: bool
    reward: float
```

### Task Definition Pattern
- **Rubric-driven**: Ground truth passed at reset()
- **Multiple finalization patterns**: FINAL(), FINAL_VAR(), dict with ready flag
- **External graders**: CustomMetricRubric for user-provided scoring

### Reward Design
- **Composable**: REPLRubric = outcome + process
- **Outcome (terminal)**: ExactMatch, FuzzyMatch, or CustomMetric
- **Process (per-step)**: +success_reward, -error_penalty
- **Failure**: -failure_reward if max_iterations without answer

### Worth Copying
1. **Composable rubrics** - outcome + process separation
2. **Recursive backend** - Protocol-based with depth limits
3. **Message-based loop** - Explicit iteration with timeout checks
4. **Variable snapshots** - Serialize namespace state
5. **Dual API** - Sync + async with same models
6. **Cooperative timeout** - perf_counter() checks, not interrupts
7. **Injected helpers** - llm_query, rlm_query available in namespace

---

## Cross-Cutting Patterns

### 1. Pydantic Models Everywhere
All environments use Pydantic BaseModel for:
- Type safety + validation
- JSON serialization
- OpenAPI schema generation
- Field descriptions for documentation

### 2. FastAPI App Factory
```python
from openenv.core.env_server.http_server import create_app

app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="my_env",
    max_concurrent_envs=1,
)
```

### 3. Client-Server Separation
- Server: Implements Environment[Action, Observation, State]
- Client: EnvClient[Action, Observation, State] wraps HTTP/WebSocket
- Local variants for in-process testing

### 4. Episode State Management
```python
class State(BaseModel):
    episode_id: str        # UUID per episode
    step_count: int        # Actions taken
    # Environment-specific metrics
```

### 5. Metadata for Flexibility
- Actions have optional `metadata: Dict[str, Any]`
- Observations include `metadata` for extra context
- Enables custom reward signals without model changes

### 6. Docker + openenv.yaml
```yaml
spec_version: 1
name: my_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

### 7. Concurrent Sessions Support
```python
class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
```

---

## Recommendations for Hackathon Project

### Use calendar_env approach if:
- Building database-backed environment (customer support, data cleaning)
- Need multi-agent evaluation isolation
- Want reusable wrapper for other MCPs

### Use reasoning_gym_env approach if:
- Simple single-step tasks (email triage, classification)
- Dataset-based evaluation
- Minimal code complexity desired

### Use tbench2_env approach if:
- Tool use benchmarking (API integration, CLI tools)
- Need Docker isolation
- Session-based interaction required

### Use carla_env approach if:
- Complex simulation with physics
- Scenario-based curriculum learning
- Trajectory-based rewards

### Use repl_env approach if:
- Code execution environment
- Recursive reasoning needed
- Composable reward functions

---

## Quick Start Checklist

For your hackathon environment, ensure:

- [ ] **3+ tasks with graders** returning scores 0.0-1.0
- [ ] **Pydantic models** for Action, Observation, State
- [ ] **openenv.yaml** with correct metadata
- [ ] **inference.py** in root (uses HF Router, not OpenAI)
- [ ] **STDOUT logging** with [START], [STEP], [END] format
- [ ] **Dockerfile** in root directory (not /server)
- [ ] **Meaningful rewards** that distinguish performance levels
- [ ] **Real-world task** with genuine value
- [ ] **< 20 min runtime** on vcpu=2, memory=8GB
- [ ] **Passes `openenv validate`**

---

## Key Files to Reference

### For Implementation Patterns:
- `calendar_env/server/openenv_wrapper/mcp_env_environment.py` - Generic wrapper
- `reasoning_gym_env/server/reasoning_gym_environment.py` - Minimal implementation
- `tbench2_env/server/tbench2_env_environment.py` - Session management
- `carla_env/server/benchmark_scenarios/base.py` - Scenario ABC
- `repl_env/rubrics.py` - Composable reward design

### For Client Usage:
- `*/client.py` - All environments have reference implementations
- `repl_env/runner.py` - Message-based orchestration loop

### For Server Setup:
- `*/server/app.py` - FastAPI app factory usage
- `*/openenv.yaml` - Configuration examples
- `*/Dockerfile` - Docker image patterns

---

## Next Steps

1. **Choose architecture**: Pick closest reference environment to your task
2. **Copy skeleton**: Use `openenv init` or copy from reference
3. **Define models**: Start with Action/Observation Pydantic models
4. **Implement graders**: 3 tasks with programmatic scoring
5. **Test locally**: Use client.py pattern for rapid iteration
6. **Validate**: Run `openenv validate` before deployment
7. **Deploy**: `openenv push` to Hugging Face Spaces
