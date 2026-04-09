"""
Antibiotic Stewardship — FastAPI backend
Fixes:
  1. Env stored per task_id (survives /reset cycling, no global single-instance race)
  2. /reset body parsed correctly with Body(...)
  3. Removed duplicate server/__init__.py (consolidate here)
  4. /step and /grade return 503 (not 400) when no session → inference.py retries it
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict

from environment import AntibioticResistanceEnv, TASKS
from models import Action, Observation, StepResult, ResetResult, GradeResult

app = FastAPI(
    title="Antibiotic Stewardship OpenEnv",
    description="RL environment for antibiotic decision-making",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── session store: keyed by task_id so sequential tasks don't clobber each other
# Using a dict means easy→medium→hard all have independent env objects.
_envs: Dict[str, AntibioticResistanceEnv] = {}
_active_task: Optional[str] = None   # track most-recent reset task


def _get_env(task_id: Optional[str] = None) -> AntibioticResistanceEnv:
    """
    Returns the env for task_id (or the most-recently reset task).
    Returns 503 instead of 400 so inference.py _call_with_retry will retry
    on transient cold-start / race conditions.
    """
    key = task_id or _active_task
    if key is None or key not in _envs:
        raise HTTPException(
            status_code=503,
            detail="No active session. Call /reset first. (Retryable)"
        )
    return _envs[key]


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Antibiotic Stewardship OpenEnv API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {
        task_id: {
            "description": cfg["description"],
            "patients": cfg["patients_per_episode"],
        }
        for task_id, cfg in TASKS.items()
    }


class ResetRequest(BaseModel):
    task_id: str = "easy"

class StepRequest(BaseModel):
    antibiotic: int
    task_id: Optional[str] = None   # optional: if provided, routes to correct env directly


@app.post("/reset")
async def reset(
    req: ResetRequest = Body(default=ResetRequest())   # FIX: explicit Body() so JSON body is always parsed
):
    global _active_task

    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{req.task_id}'. Choose from: {list(TASKS)}")

    env = AntibioticResistanceEnv(task_id=req.task_id)
    obs_dict = env.reset()

    _envs[req.task_id] = env          # store by task_id — doesn't clobber other tasks
    _active_task = req.task_id

    return ResetResult(
        observation=Observation(**obs_dict),
        task_id=req.task_id,
        task_description=TASKS[req.task_id]["description"],
    )


@app.post("/step")
def step(req: StepRequest):
    # prefer explicit task_id; fall back to _active_task for backward compat
    env = _get_env(req.task_id)
    obs_dict, reward, done, info = env.step(req.antibiotic)
    obs = Observation(**obs_dict) if obs_dict is not None else None
    return StepResult(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state():
    return _get_env().get_state()


@app.get("/grade")
def grade(task_id: Optional[str] = None):
    # prefer explicit task_id query param; fall back to _active_task
    env = _get_env(task_id)
    return GradeResult(**env.grade())


# ── entry ─────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
