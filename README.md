---
title: Antibiotic Stewardship OpenEnv
emoji: 🦠
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Antibiotic Stewardship — OpenEnv Submission

An RL environment where an LLM agent acts as a clinical pharmacist, treating
patients while managing antimicrobial resistance (AMR). AMR is a WHO-priority
global health crisis responsible for ~700 000 deaths/year.

## Tasks

| Task   | Patients | Difficulty |
|--------|----------|------------|
| easy   | 10       | Mild infections, healthy adults, low resistance |
| medium | 15       | Mixed severity, elderly/paediatric, some resistance |
| hard   | 20       | Emergency ward, MRSA, high resistance, vulnerable patients |

## Antibiotics

| ID | Drug          | Use case          |
|----|---------------|-------------------|
| 0  | Penicillin    | Mild infections   |
| 1  | Azithromycin  | Moderate          |
| 2  | Vancomycin    | Severe / MRSA     |

## API

| Endpoint      | Method | Description              |
|---------------|--------|--------------------------|
| `/reset`      | POST   | Start episode `{task_id}`|
| `/step`       | POST   | Take action `{antibiotic}`|
| `/state`      | GET    | Full environment state   |
| `/grade`      | GET    | Score 0.0–1.0            |
| `/tasks`      | GET    | List available tasks     |
| `/health`     | GET    | Health check             |

## Quick start

```bash
# Start server
docker build -t abx-env . && docker run -p 7860:7860 abx-env

# Reset to easy task
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
     -d '{"task_id": "easy"}'

# Take a step (give Azithromycin)
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
     -d '{"antibiotic": 1}'

# Get score
curl http://localhost:7860/grade
```

## Run inference

```bash
export API_BASE_URL=https://YOUR-SPACE.hf.space
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
python inference.py
```

## Scoring

`score = max(0, total_reward) / (patients × 10)`

Rewards per patient:
- **+10** — correct drug, patient cured
- **+3**  — correct drug, partial cure (resistance 40–70%)
- **−5 to −15** — wrong drug or treatment failure
- **−12** — Vancomycin overkill on mild case
- **side-effect penalty** — stronger drugs on elderly/children
