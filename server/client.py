import requests
from typing import Optional, Dict, Any
from models import Action, Observation, StepResult, ResetResult, GradeResult

class AntibioticStewardshipClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy") -> ResetResult:
        resp = requests.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=60)
        resp.raise_for_status()
        return ResetResult(**resp.json())

    def step(self, action: Action) -> StepResult:
        resp = requests.post(f"{self.base_url}/step", json=action.model_dump(), timeout=60)
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/state", timeout=60)
        resp.raise_for_status()
        return resp.json()

    def grade(self, task_id: Optional[str] = None) -> GradeResult:
        params = {"task_id": task_id} if task_id else {}
        resp = requests.get(f"{self.base_url}/grade", params=params, timeout=60)
        resp.raise_for_status()
        return GradeResult(**resp.json())
