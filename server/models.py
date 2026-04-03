from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class Action(BaseModel):
    antibiotic: int = Field(..., ge=0, le=2, description="0=Penicillin, 1=Azithromycin, 2=Vancomycin")


class Observation(BaseModel):
    infection: str = Field(..., description="Infection type: E.coli, Staph, Strep, MRSA")
    severity: int = Field(..., ge=1, le=3, description="1=mild, 2=moderate, 3=severe")
    age: int = Field(..., ge=1, le=100, description="Patient age in years")
    resistance: Dict[str, float] = Field(..., description="Resistance levels for each antibiotic (0.0-1.0)")
    patients_treated: int = Field(..., description="Patients treated so far in episode")
    patients_total: int = Field(..., description="Total patients in episode")


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class ResetResult(BaseModel):
    observation: Observation
    task_id: str
    task_description: str


class GradeResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    raw_score: float
    max_score: float
    breakdown: Dict[str, Any] = {}
