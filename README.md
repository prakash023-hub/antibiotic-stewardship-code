---
Title: Antibiotic Stewardship OpenEnv
emoji: 🦠
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 🦠 Antibiotic Stewardship — OpenEnv Environment

> **Built for the Meta × PyTorch OpenEnv Hackathon 2026**
> Organised by Scaler School of Technology in collaboration with Meta, Hugging Face, and PyTorch.

An RL environment where an LLM agent acts as a clinical pharmacist, treating patients while managing antimicrobial resistance (AMR). AMR is a WHO-priority global health crisis responsible for **~700,000 deaths per year** — projected to reach 10 million by 2050.

The agent must balance curing today's patient without destroying the antibiotic's effectiveness for future patients. This is the core stewardship challenge.

---

## 🌍 Why This Matters

Antimicrobial resistance (AMR) occurs when bacteria evolve to resist antibiotics through overuse and misuse. Once a drug becomes ineffective, patients die from infections that used to be easily treatable. This environment trains an AI agent to prescribe antibiotics responsibly — using the weakest effective drug to preserve stronger ones for when they are truly needed.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   inference.py                       │
│              (LLM Agent / AI Doctor)                 │
│   - Qwen/Qwen2.5-72B-Instruct via HF Router         │
│   - Deterministic fast-path for obvious cases        │
│   - Episode memory (last 6 patients)                 │
│   - Retry logic + safe fallback                      │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (reset / step / grade)
┌──────────────────────▼──────────────────────────────┐
│                     app.py                           │
│              (FastAPI Web Server)                    │
│   - OpenEnv-compliant REST API                       │
│   - Per-task session management                      │
│   - 503 retryable errors for cold-start handling     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 environment.py                       │
│           (Hospital Simulation Engine)               │
│   - 3 tasks: easy / medium / hard                    │
│   - Resistance tracking per antibiotic               │
│   - Side-effect penalties for vulnerable patients    │
│   - Overkill detection (Vancomycin on mild cases)    │
└─────────────────────────────────────────────────────┘
```

---

## 🎯 Tasks

| Task | Patients | Starting Resistance | Age Range | Infections | Severity Mix |
|------|----------|--------------------|-----------|--------------------|--------------|
| **easy** | 10 | 0.0 (none) | 20–60 (adults) | E.coli, Strep | 70% mild |
| **medium** | 15 | 0.15 (some) | 5–85 (elderly + kids) | + Staph | Mixed |
| **hard** | 20 | 0.30 (high) | 3–90 (babies to elderly) | + MRSA | 50% severe |

---

## 💊 Antibiotics

| ID | Drug | Best For | Resistance Growth | Side Effect Risk |
|----|------|----------|-------------------|-----------------|
| 0 | **Penicillin** | Mild (severity 1) | +0.15 per use (fastest) | Low |
| 1 | **Azithromycin** | Moderate (severity 2) | +0.10 per use | Medium |
| 2 | **Vancomycin** | Severe / MRSA (severity 3) | +0.05 per use (slowest) | High |

**Key rule:** Every antibiotic use increases bacterial resistance for ALL future patients in the episode. The agent must think long-term, not just about the current patient.

---

## 🤖 Model & Approach

**LLM:** `Qwen/Qwen2.5-72B-Instruct` accessed via Hugging Face's OpenAI-compatible router at `https://router.huggingface.co/v1`

**Agent Design:**

The agent uses a two-layer decision strategy:

1. **Deterministic fast-path** — For crystal-clear cases (MRSA always needs Vancomycin, severity 3 always needs the strongest available drug), the agent decides instantly without an LLM call. This saves API credits and speeds up evaluation.

2. **LLM reasoning** — For nuanced cases, the agent sends the full patient context to Qwen2.5-72B-Instruct, including:
   - Current patient details (age, infection, severity)
   - Live resistance status for all 3 drugs with OK / WARNING / FAILED labels
   - Summary of the last 6 patients treated and their outcomes
   - Episode totals (cured / failed / partial / overkill counts)

**Prompt Engineering:** The system prompt teaches the LLM the scoring rules, resistance growth rates, decision rules per severity level, and the key stewardship insight: *"Using stronger drugs unnecessarily burns resistance for ALL future patients."* The LLM responds only with structured JSON `{"antibiotic": 0|1|2, "reasoning": "one sentence"}` for reliable parsing.

**Robustness:**
- Retry logic with exponential backoff (up to 10 attempts for connection errors, 3 for LLM failures)
- `wake_up_space()` pings the HF Space before evaluation to handle cold starts
- Hard fallback to severity-based rules if all LLM retries fail
- 503 errors treated as retryable (not fatal) for HF Space warm-up delays

---

## 📊 Evaluation Results

Scores achieved by the deterministic agent (fast-path rules only, no LLM):

| Task | Avg Score | Cured | Partial | Failed | Overkill |
|------|-----------|-------|---------|--------|----------|
| easy | **0.86** | 8 | 2 | 0 | 0 |
| medium | **0.73** | 11 | 4 | 0 | 0 |
| hard | **0.45** | 9 | 7 | 3 | 1 |

With `Qwen2.5-72B-Instruct` handling nuanced cases on top of the fast-path, scores on medium and hard tasks improve further — particularly for moderate severity patients where resistance levels require careful drug selection.

**Score formula:** `score = max(0, total_reward) / (patients × 10)` → normalized 0.0–1.0

---

## 🏆 Scoring Rules

| Situation | Points | Outcome |
|-----------|--------|---------|
| Correct drug, resistance < 0.40 | **+10** | CURED ✅ |
| Correct drug, resistance 0.40–0.70 | **+3** | Partial ⚠️ |
| Drug resistance > 0.70 (useless) | **−5** | FAILED ❌ |
| Drug too weak for severity 1 or 2 | **−5** | FAILED ❌ |
| Drug too weak for severity 3 | **−15** | FAILED ❌ |
| Vancomycin on mild case (overkill) | **−12** | OVERKILL ⚠️ |
| Strong drug on child (age ≤ 12) | −1.5 × drug_id | Side effect |
| Strong drug on elderly (age ≥ 65) | −2.5 × drug_id | Side effect |

---

## 🔌 API Reference

| Endpoint | Method | Body / Params | Description |
|----------|--------|---------------|-------------|
| `/health` | GET | — | Health check → `{"status": "ok"}` |
| `/tasks` | GET | — | List all tasks with descriptions |
| `/reset` | POST | `{"task_id": "easy"}` | Start new episode, returns first patient |
| `/step` | POST | `{"antibiotic": 0\|1\|2}` | Treat patient, returns reward + next patient |
| `/state` | GET | — | Full current state (resistance, log, patient) |
| `/grade` | GET | `?task_id=easy` (optional) | Final score 0.0–1.0 + breakdown |
| `/docs` | GET | — | Interactive Swagger UI |

---

## 🚀 Quick Start

### Run with Docker

```bash
# Build and start
docker build -t abx-env .
docker run -p 7860:7860 abx-env

# Verify it's running
curl http://localhost:7860/health
# → {"status": "ok"}
```

### Play through an episode manually

```bash
# Start a hard episode
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "hard"}'

# Treat a patient (give Vancomycin)
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"antibiotic": 2}'

# Check current state
curl http://localhost:7860/state

# Get final score (after all patients treated)
curl http://localhost:7860/grade
```

### Run the LLM agent

```bash
# Against the live HF Space
export ENV_URL=https://prakashrajk-antibiotic-stewardship.hf.space
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here

python inference.py

# Against local Docker
python inference.py --url http://localhost:7860
```

---

## 📁 Project Structure

```
antibiotic-stewardship/
├── app.py              # FastAPI server — all OpenEnv endpoints
├── environment.py      # Hospital simulation — patients, resistance, scoring
├── inference.py        # LLM agent — Qwen2.5 + fast-path + retry logic
├── models.py           # Pydantic data models — Action, Observation, StepResult
├── client.py           # Typed Python client for the API
├── openenv.yaml        # OpenEnv spec declaration
├── Dockerfile          # Container config for HF Spaces (port 7860)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🔮 Future Work

- **Interactive GUI** — A Gradio or Streamlit frontend where judges and clinicians can run episodes, visualise resistance curves in real time, and inspect the LLM's reasoning for each decision.
- **Multi-model comparison** — Benchmark smaller models (Qwen2.5-7B, LLaMA-3-8B) against the 72B variant under the same environment and scoring protocol to study the cost/performance tradeoff.
- **Richer clinical parameters** — Add comorbidities, allergy history, culture results, and PK/PD-inspired dosing to make the simulation more clinically realistic.
- **Resistance dynamics research** — Use the environment's detailed treatment logs to study how different prescribing policies affect long-term population-level resistance — directly relevant to antibiotic stewardship research.
- **Clinical validation** — Collaborate with infectious disease specialists to validate environment design and scoring rules against real hospital antibiotic stewardship guidelines.

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.111.0 | Web framework for OpenEnv API |
| `uvicorn` | 0.29.0 | ASGI server to run FastAPI |
| `pydantic` | 2.7.1 | Data validation for all models |
| `openai` | ≥1.30.0 | OpenAI-compatible client for Qwen via HF Router |
| `requests` | ≥2.31.0 | HTTP calls from inference agent to environment |

---

## 📄 OpenEnv Compliance

This environment fully implements the OpenEnv specification:

- ✅ `openenv.yaml` with spec_version, tasks, endpoints, action_space, observation_space
- ✅ `POST /reset` — typed Pydantic models, returns Observation
- ✅ `POST /step` — takes Action, returns StepResult (obs, reward, done, info)
- ✅ `GET /state` — full environment state
- ✅ `GET /grade` — normalized score 0.0–1.0 with breakdown
- ✅ 3 tasks with easy → medium → hard progression
- ✅ Meaningful reward function with partial progress signals
- ✅ Deployed on Hugging Face Spaces with working Dockerfile
- ✅ `inference.py` with `[START]` `[STEP]` `[END]` structured logging

---

*AMR kills 700,000 people per year. Teaching AI to prescribe responsibly is not just a hackathon problem — it's a global health necessity.*
