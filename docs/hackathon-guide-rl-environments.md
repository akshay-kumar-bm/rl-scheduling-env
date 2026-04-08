# Building RL Environments for Hackathon: Complete Guide

## Overview
This guide provides comprehensive insights for building real-world Reinforcement Learning (RL) environments using the OpenM (Open Environment) library for hackathon participation.

---

## 1. Fundamentals of Reinforcement Learning

### The Mechanism
- **How it Works:** Model generates candidate implementations (actions) → Environment verifies/tests → Environment provides reward signal (score) based on pre-defined rubrics
- **Purpose:** Tells the model what is good or bad through trial and error rather than long-context prompts

### Position in Training Pipeline
- Typically follows **Supervised Fine-Tuning (SFT)**
- Used to "squeeze out" final performance gains on specific capabilities
- More efficient alternative to "in-context learning" (which degrades with longer prompts)

### Key Challenges

#### Reward Hacking
- Models learn to "game" the verifier to get high scores without actually solving the task
- **Mitigation:** Inspect output trajectories or use multiple reward functions

#### Curriculum Learning
- Start with easy tasks and build complexity progressively
- Ensures model receives consistent reward signal
- Prevents "wasting compute" on tasks that are too difficult initially

---

## 2. Introduction to OpenM

### What is OpenM?
- Collaborative project between Meta, Hugging Face, and others
- Standardizes RL environments (like Hugging Face standardized language models)
- Single, consistent API for environments
- Interoperable with training frameworks (TRL, Unsloth, etc.)

### Core Components
Standard OpenM environment requires defining:
- **Actions** (as Pydantic objects)
- **Observations** (as Pydantic objects)
- **States** (as Pydantic objects)

---

## 3. Technical Implementation

### CLI Workflow
```bash
# Initialize skeleton environment
openm init

# Validate setup
openm validate

# Deploy to Hugging Face Spaces
openm push
```

### Agent Integration
- Use coding agents (like Codeex) with OpenM "skills"
- Automatically generate environment code from prompts

### Deployment
- Environments deployed as Docker containers on Hugging Face
- Provides web interface for manual testing and debugging
- **Important:** Dockerfile must be moved outside `/server` folder to main project directory

---

## 4. Hackathon Requirements

### Environment Quality

#### Real-World Focus (Critical)
- **Must build:** Real-world task environments (healthcare, email triage, code optimization)
- **Avoid:** "Toy" environments, games (Wordle, Connect 4, etc.)
- **Goal:** Environment that could realistically be used in model's post-training RL run

#### Complexity Requirements
- Map **long-running tasks** with multiple trajectories/routes
- Agent should have various possible approaches to solve the task

### Technical Requirements

#### Mandatory Inference Script
- **Required for every submission**
- Used by organizers to evaluate environment effectiveness
- Measures how well environment provides rewards to model

#### API Configuration
- **No OpenAI API key required**
- Use **Hugging Face token** instead
- Use provided **HF Router** (API base URL) for model calls
- HF Router handles model calls through Hugging Face

#### Docker Setup
- Move Dockerfile outside `/server` folder to main project directory
- Run `openm validate` before submission

### Reward Signal Design

#### Requirements
- Score typically between 0 and 1
- Must deliver valid signal indicating "good" or "bad" performance
- **Grading Diversity:** Must not return same score every time
- Should distinguish between different performance levels

#### Best Practices
- Start with achievable tasks for the model
- Ensure task is feasible but challenging
- Avoid tasks too difficult or out-of-distribution for the model

---

## 5. Grading Criteria

Evaluation based on:

1. **Utility of the Idea**
   - How useful is the task for real-world AI?
   - Does it represent authentic human tasks?

2. **Quality of the Grader**
   - Returns diverse scores (not same score every time)
   - Value between 0 and 1
   - Distinguishes performance levels

3. **Technical Design**
   - Environment architecture and implementation
   - Successful execution of inference script

4. **Novelty**
   - Key criterion for high scores
   - Create something not thought of yet
   - Solve problems in unique domains
   - **Plagiarism is strictly prohibited**

---

## 6. Submission Guidelines

### Deadline
- **Round One:** April 8th

### Submission Process
- Push environment to **Hugging Face Spaces** using `openm push`
- Submit URL of Hugging Face Space
- Multiple submissions allowed (latest accurate submission used)

### Collaboration
- Teams are **highly encouraged**
- Helps manage technical and creative requirements

---

## 7. High-Value Environment Ideas

### Healthcare Domain
- Medical triage tools
- Navigating medical records
- Healthcare-specific software tool utilization

### Productivity and Operations
- **Email Triage:** Prioritize, categorize, respond to complex inbox
- **Calendar Management:** Coordinate schedules, handle conflicts across multiple participants

### Technical and Code Optimization
- **Kernel Optimization:** Benchmark and optimize PyTorch/GPU kernels for speed and efficiency
- **Repository Maintenance:** Navigate GitHub to identify/fix bugs, run test suites

### Logistics and Travel
- **Complex Flight Booking:** Navigate changing availability, multi-leg transfers, request missing information from users

### API and Tool Integration
- Wide set of real-world tools
- Interactive APIs that agents must learn to use correctly

---

## 8. Best Practices Summary

### Do's
- Focus on real-world utility
- Design long-running, multi-trajectory tasks
- Implement diverse grading systems
- Start with curriculum learning approach
- Validate thoroughly before submission
- Work in teams for better results
- Aim for novelty and uniqueness

### Don'ts
- Avoid toy environments or games
- Don't create tasks too difficult for models
- Don't implement single-score graders
- Avoid plagiarism
- Don't submit without testing inference script
- Don't use tasks without clear reward signals

---

## 9. Technical Checklist

- [ ] Initialize project with `openm init`
- [ ] Define Actions, Observations, States as Pydantic objects
- [ ] Implement diverse reward function (0-1 range)
- [ ] Create mandatory inference script
- [ ] Configure HF token and router (not OpenAI key)
- [ ] Move Dockerfile to main directory (outside /server)
- [ ] Run `openm validate` to verify setup
- [ ] Test environment locally
- [ ] Deploy with `openm push` to Hugging Face Spaces
- [ ] Submit Hugging Face Space URL before April 8th

---

## Resources

- **OpenM Library:** Standardized RL environment framework
- **Hugging Face Spaces:** Deployment platform
- **HF Router:** API for model access
- **Training Frameworks:** TRL, Unsloth (compatible with OpenM)

---

*This guide synthesizes best practices for building competitive RL environments for hackathons. Focus on real-world utility, technical excellence, and novel approaches for the best results.*
