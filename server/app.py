from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from environment import AntibioticResistanceEnv, TASKS
from models import Action, Observation, StepResult, ResetResult, GradeResult

app = FastAPI(title="Antibiotic Stewardship OpenEnv", description="RL environment", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env = None

def _get_env():
    if _env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    return _env

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks():
    return {task_id: {"description": cfg["description"], "patients": cfg["patients_per_episode"]} for task_id, cfg in TASKS.items()}

class ResetRequest(BaseModel):
    task_id: str = "easy"

@app.post("/reset")
async def reset(req: ResetRequest = None):
    global _env
    if req is None:
        req = ResetRequest()
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task")
    _env = AntibioticResistanceEnv(task_id=req.task_id)
    obs_dict = _env.reset()
    from models import ResetResult, Observation
    return ResetResult(observation=Observation(**obs_dict), task_id=req.task_id, task_description=TASKS[req.task_id]["description"])

@app.post("/step")
def step(action: Action):
    env = _get_env()
    obs_dict, reward, done, info = env.step(action.antibiotic)
    obs = Observation(**obs_dict) if obs_dict else None
    return StepResult(observation=obs, reward=reward, done=done, info=info)

@app.get("/state")
def state():
    return _get_env().get_state()

@app.get("/grade")
def grade():
    return GradeResult(**_get_env().grade())

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
