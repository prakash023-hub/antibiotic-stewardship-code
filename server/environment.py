"""
Antibiotic Stewardship Environment — 3 tasks
Task difficulty is controlled via patient mix, resistance pressure, and scoring.
"""

import random
import uuid
from typing import Optional, Tuple, Dict, Any

# ── antibiotic constants ──────────────────────────────────────────────────────
ANTIBIOTIC_NAMES = {0: "Penicillin", 1: "Azithromycin", 2: "Vancomycin"}
RESISTANCE_RATES  = {0: 0.15, 1: 0.10, 2: 0.05}
MIN_STRENGTH_FOR_SEVERITY = {1: 0, 2: 1, 3: 2}

# ── task definitions ──────────────────────────────────────────────────────────
TASKS = {
    "easy": {
        "description": (
            "Treat 10 patients with mild-to-moderate infections. "
            "Resistance starts low and patients are all healthy adults. "
            "Goal: match antibiotic strength to severity and avoid over-prescribing."
        ),
        "patients_per_episode": 10,
        "severity_weights": [0.7, 0.3, 0.0],   # mostly mild
        "infection_pool": ["E.coli", "Strep"],
        "age_range": (20, 60),
        "initial_resistance": 0.0,
        "max_score_per_patient": 10.0,
    },
    "medium": {
        "description": (
            "Treat 15 patients with mixed severities. "
            "Some resistance has already developed. "
            "Includes elderly and paediatric patients who are sensitive to side-effects."
        ),
        "patients_per_episode": 15,
        "severity_weights": [0.35, 0.40, 0.25],
        "infection_pool": ["E.coli", "Staph", "Strep"],
        "age_range": (5, 85),
        "initial_resistance": 0.15,
        "max_score_per_patient": 10.0,
    },
    "hard": {
        "description": (
            "Emergency ward: 20 patients including MRSA cases, severe infections, "
            "high pre-existing resistance, and a high proportion of vulnerable patients "
            "(elderly and paediatric). Every decision has a significant consequence."
        ),
        "patients_per_episode": 20,
        "severity_weights": [0.15, 0.35, 0.50],  # mostly severe
        "infection_pool": ["E.coli", "Staph", "Strep", "MRSA"],
        "age_range": (3, 90),
        "initial_resistance": 0.30,
        "max_score_per_patient": 10.0,
    },
}


class AntibioticResistanceEnv:
    def __init__(self, task_id: str = "easy"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task '{task_id}'. Choose from: {list(TASKS)}")
        self.task_id   = task_id
        self.task_cfg  = TASKS[task_id]
        self.session_id: Optional[str] = None
        # internal state (populated by reset)
        self.patients_treated   = 0
        self.resistance_levels  : Dict[int, float] = {}
        self.total_score        = 0.0
        self.episode_complete   = False
        self.treatment_log      : list = []
        self.current_patient    : Optional[Dict] = None

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        cfg = self.task_cfg
        self.session_id       = str(uuid.uuid4())
        self.patients_treated = 0
        self.total_score      = 0.0
        self.episode_complete = False
        self.treatment_log    = []
        r0 = cfg["initial_resistance"]
        self.resistance_levels = {0: r0, 1: r0 * 0.7, 2: r0 * 0.3}
        self.current_patient  = self._generate_patient()
        return self._get_observation()

    def step(self, antibiotic: int) -> Tuple[Optional[Dict], float, bool, Dict]:
        if self.episode_complete:
            return None, 0.0, True, {"error": "episode already complete"}

        reward, outcome, side_fx = self._evaluate_treatment(antibiotic)

        # update resistance
        self.resistance_levels[antibiotic] = min(
            1.0, self.resistance_levels[antibiotic] + RESISTANCE_RATES[antibiotic]
        )
        self.total_score      += reward
        self.patients_treated += 1

        self.treatment_log.append({
            "patient_num" : self.patients_treated,
            "severity"    : self.current_patient["severity"],
            "age"         : self.current_patient["age"],
            "infection"   : self.current_patient["infection"],
            "antibiotic"  : ANTIBIOTIC_NAMES[antibiotic],
            "outcome"     : outcome,
            "reward"      : reward,
            "side_effects": side_fx,
        })

        self.episode_complete = self.patients_treated >= self.task_cfg["patients_per_episode"]
        if not self.episode_complete:
            self.current_patient = self._generate_patient()

        obs  = self._get_observation() if not self.episode_complete else None
        info = {"outcome": outcome, "side_effects": side_fx, "total_score": self.total_score}
        return obs, reward, self.episode_complete, info

    def get_state(self) -> Dict[str, Any]:
        return {
            "task_id"          : self.task_id,
            "session_id"       : self.session_id,
            "patients_treated" : self.patients_treated,
            "patients_total"   : self.task_cfg["patients_per_episode"],
            "resistance_levels": {ANTIBIOTIC_NAMES[k]: round(v, 3)
                                  for k, v in self.resistance_levels.items()},
            "total_score"      : round(self.total_score, 2),
            "episode_complete" : self.episode_complete,
            "treatment_log"    : self.treatment_log,
            "current_patient"  : self.current_patient,
        }

    def grade(self) -> Dict[str, Any]:
        """Return a score in [0.0, 1.0] based on clinical quality."""
        n   = self.task_cfg["patients_per_episode"]
        max_possible = n * self.task_cfg["max_score_per_patient"]  # 10 per patient

        # Penalise incomplete episodes
        treated = self.patients_treated
        if treated == 0:
            return {"score": 0.0, "raw_score": 0.0, "max_score": max_possible, "breakdown": {}}

        # Outcome counts
        cured    = sum(1 for t in self.treatment_log if t["outcome"] == "CURED")
        failed   = sum(1 for t in self.treatment_log if t["outcome"] == "FAILED")
        overkill = sum(1 for t in self.treatment_log if t["outcome"] == "OVERKILL")
        partial  = sum(1 for t in self.treatment_log if t["outcome"] == "Partial")

        raw        = self.total_score
        normalised = max(0.0, min(1.0, raw / max_possible))

        return {
            "score"    : round(normalised, 4),
            "raw_score": round(raw, 2),
            "max_score": max_possible,
            "breakdown": {
                "cured"   : cured,
                "failed"  : failed,
                "overkill": overkill,
                "partial" : partial,
                "avg_reward_per_patient": round(raw / treated, 2),
            },
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _generate_patient(self) -> Dict:
        cfg        = self.task_cfg
        severities = [1, 2, 3]
        severity   = random.choices(severities, weights=cfg["severity_weights"])[0]
        age_lo, age_hi = cfg["age_range"]
        return {
            "infection": random.choice(cfg["infection_pool"]),
            "severity" : severity,
            "age"      : random.randint(age_lo, age_hi),
        }

    def _get_observation(self) -> Dict:
        p = self.current_patient
        return {
            "infection"       : p["infection"],
            "severity"        : p["severity"],
            "age"             : p["age"],
            "resistance"      : {str(k): round(v, 3) for k, v in self.resistance_levels.items()},
            "patients_treated": self.patients_treated,
            "patients_total"  : self.task_cfg["patients_per_episode"],
        }

    def _evaluate_treatment(self, drug_id: int) -> Tuple[float, str, float]:
        severity   = self.current_patient["severity"]
        age        = self.current_patient["age"]
        resistance = self.resistance_levels[drug_id]
        required   = MIN_STRENGTH_FOR_SEVERITY[severity]
        side_fx    = 0.0

        # Under-powered drug
        if drug_id < required:
            return (-15.0 if severity == 3 else -5.0), "FAILED", 0.0

        # Drug no longer effective
        if resistance > 0.70:
            return -5.0, "FAILED", 0.0

        # Partial vs full cure
        if resistance > 0.40:
            base_score = 3.0
            outcome    = "Partial"
        else:
            base_score = 10.0
            outcome    = "CURED"

        # Vulnerable-population side-effects
        if age <= 12 and drug_id >= 1:
            side_fx -= drug_id * 1.5
        elif age >= 65:
            side_fx -= drug_id * 2.5

        # Stewardship: Vancomycin on mild = overkill
        if drug_id == 2 and severity == 1:
            base_score -= 12.0
            outcome     = "OVERKILL"

        return base_score + side_fx, outcome, side_fx
