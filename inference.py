import os, json, argparse, requests
from openai import OpenAI
from typing import Optional
from uuid import uuid4

# ── config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(api_key=OPENAI_API_KEY)

ANTIBIOTIC_NAMES  = {0: "Penicillin", 1: "Azithromycin", 2: "Vancomycin"}
RESISTANCE_DANGER = 0.70
RESISTANCE_WARN   = 0.50


# ── environment API ───────────────────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()

def env_step(antibiotic: int) -> dict:
    r = requests.post(f"{API_BASE_URL}/step", json={"antibiotic": antibiotic})
    r.raise_for_status()
    return r.json()

def env_grade() -> dict:
    r = requests.get(f"{API_BASE_URL}/grade")
    r.raise_for_status()
    return r.json()


# ── deterministic fast-path ───────────────────────────────────────────────────

def deterministic_choice(obs: dict) -> Optional[int]:
    """
    Returns an antibiotic (0/1/2) for crystal-clear cases,
    or None to let the LLM decide.
    """
    severity   = obs["severity"]
    infection  = obs["infection"]
    age        = obs["age"]
    resistance = {int(k): float(v) for k, v in obs["resistance"].items()}

    def usable(drug_id: int) -> bool:
        return resistance[drug_id] <= RESISTANCE_DANGER

    # MRSA always needs Vancomycin
    if infection == "MRSA":
        return 2

    # Severe → always Vancomycin (cascade down only if completely failed)
    if severity == 3:
        if usable(2): return 2
        if usable(1): return 1
        return 0

    # Vulnerable + mild + Penicillin still works → never escalate
    is_vulnerable = age <= 12 or age >= 65
    if is_vulnerable and severity == 1 and usable(0):
        return 0

    # All options failed for this severity → must escalate
    if severity == 1 and not usable(0) and not usable(1):
        return 2
    if severity == 2 and not usable(1) and not usable(0):
        return 2

    return None   # LLM handles the nuanced middle-ground


# ── resistance commentary ─────────────────────────────────────────────────────

def build_resistance_commentary(obs: dict) -> str:
    resistance = {int(k): float(v) for k, v in obs["resistance"].items()}
    lines = []
    for drug_id, level in sorted(resistance.items()):
        name = ANTIBIOTIC_NAMES[drug_id]
        if level > RESISTANCE_DANGER:
            lines.append(f"  FAILED   {name} (drug {drug_id}): {level:.2f} — do NOT use")
        elif level > RESISTANCE_WARN:
            lines.append(f"  WARNING  {name} (drug {drug_id}): {level:.2f} — high risk, prefer alternative")
        else:
            lines.append(f"  OK       {name} (drug {drug_id}): {level:.2f} — safe to use")
    return "\n".join(lines)


# ── episode memory ────────────────────────────────────────────────────────────

def build_history_summary(history: list) -> str:
    if not history:
        return "No patients treated yet."

    lines = ["Recent patients (last 6):"]
    for h in history[-6:]:
        emoji = {"CURED": "[OK]", "Partial": "[~~]", "FAILED": "[XX]", "OVERKILL": "[!!]"}.get(h["outcome"], "[?]")
        lines.append(
            f"  {emoji} Patient {h['patient_num']}: "
            f"age={h['age']}, sev={h['severity']}, {h['infection']}, "
            f"gave {h['drug_name']} -> {h['outcome']} ({h['reward']:+.1f})"
        )

    outcomes = [h["outcome"] for h in history]
    lines.append(
        f"\nTotals so far: "
        f"{outcomes.count('CURED')} cured, "
        f"{outcomes.count('FAILED')} failed, "
        f"{outcomes.count('Partial')} partial, "
        f"{outcomes.count('OVERKILL')} overkill"
    )
    return "\n".join(lines)


# ── LLM agent ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a world-class clinical pharmacist specialising in antibiotic stewardship.
Your goal: maximise patient outcomes while preserving antibiotic effectiveness across the WHOLE episode.

ANTIBIOTICS:
  0 = Penicillin    — mild infections. Resistance grows +0.15 per use (fastest).
  1 = Azithromycin  — moderate infections. Resistance grows +0.10 per use.
  2 = Vancomycin    — severe/MRSA. Resistance grows only +0.05 per use (slowest).

SCORING:
  +10  correct drug, patient cured (resistance < 0.40)
  +3   correct drug, partial cure  (resistance 0.40-0.70)
  -5   drug failed (resistance > 0.70 OR underpowered)
  -15  underpowered for severity 3
  -12  Vancomycin overkill on severity 1
  Side-effect penalty: age<=12 -> -1.5 x drug_id; age>=65 -> -2.5 x drug_id

DECISION RULES:
  severity 1 (mild):
    - prefer Penicillin (0); switch to Azithromycin (1) if resistance[0] > 0.50
    - NEVER use Vancomycin (2) on mild — it loses 12 points
  severity 2 (moderate):
    - prefer Azithromycin (1); escalate to Vancomycin (2) only if resistance[1] > 0.60
  severity 3 (severe):
    - always Vancomycin (2) unless resistance[2] > 0.70
  MRSA:
    - always Vancomycin (2)
  Vulnerable patients (age<=12 or age>=65):
    - use weakest drug that still works — stronger drugs cause side-effect penalties

KEY INSIGHT: Using stronger drugs unnecessarily burns resistance for ALL future patients.
Think across the whole episode, not just the current patient.

Respond ONLY with valid JSON: {"antibiotic": <0|1|2>, "reasoning": "<one sentence>"}"""


def ask_llm(obs: dict, history: list, retries: int = 3) -> int:
    user_msg = f"""CURRENT PATIENT:
{json.dumps(obs, indent=2)}

LIVE RESISTANCE STATUS:
{build_resistance_commentary(obs)}

EPISODE HISTORY:
{build_history_summary(history)}

This is patient {obs['patients_treated'] + 1} of {obs['patients_total']}.
Choose the antibiotic:"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result     = json.loads(response.choices[0].message.content)
            antibiotic = int(result["antibiotic"])
            # reasoning  = result.get("reasoning", "")

            if antibiotic not in (0, 1, 2):
                raise ValueError(f"Invalid choice: {antibiotic}")

            # old human log:
            # print(f"         LLM -> {ANTIBIOTIC_NAMES[antibiotic]}: {reasoning}")
            return antibiotic

        except Exception as e:
            # print(f"         LLM attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                break

    # Hard fallback
    fallback = {1: 0, 2: 1, 3: 2}.get(obs.get("severity", 2), 1)
    # print(f"         Fallback -> {ANTIBIOTIC_NAMES[fallback]}")
    return fallback


# ── episode runner ────────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    # run/task banners (safe, but evaluator ignores them)
    # print(f"\n{'='*60}")
    # print(f"  TASK: {task_id.upper()}")
    # print(f"{'='*60}")

    reset_data  = env_reset(task_id)
    observation = reset_data["observation"]
    history     = []

    run_id = str(uuid4())
    total_patients = observation.get("patients_total", 0)

    # [START] log
    start_payload = {
        "run_id": run_id,
        "task_id": task_id,
        "total_patients": total_patients,
    }
    print(f"[START] {json.dumps(start_payload)}", flush=True)

    while True:
        sev       = observation["severity"]
        infection = observation["infection"]
        age       = observation["age"]
        patient_n = observation["patients_treated"] + 1
        total     = observation["patients_total"]

        # old per-patient human print:
        # print(f"\n  [Patient {patient_n}/{total}] age={age}, sev={sev}, {infection}")

        fast = deterministic_choice(observation)
        if fast is not None:
            antibiotic = fast
            decision_source = "fast_path"
        else:
            antibiotic = ask_llm(observation, history)
            decision_source = "llm"

        step_result = env_step(antibiotic)
        reward  = step_result["reward"]
        done    = step_result["done"]
        info    = step_result.get("info", {})
        outcome = info.get("outcome", "?")

        history.append({
            "patient_num": patient_n,
            "severity"   : sev,
            "age"        : age,
            "infection"  : infection,
            "drug_id"    : antibiotic,
            "drug_name"  : ANTIBIOTIC_NAMES[antibiotic],
            "outcome"    : outcome,
            "reward"     : reward,
        })

        # [STEP] log
        step_payload = {
            "run_id": run_id,
            "task_id": task_id,
            "patient_index": patient_n,
            "state": {
                "severity": sev,
                "infection": infection,
                "age": age,
                "patients_treated": observation["patients_treated"],
                "patients_total": total,
                "resistance": observation["resistance"],
            },
            "action": {
                "antibiotic_id": antibiotic,
                "antibiotic_name": ANTIBIOTIC_NAMES[antibiotic],
                "source": decision_source,
            },
            "reward": reward,
            "done": done,
            "info": {
                "outcome": outcome,
            },
        }
        print(f"[STEP] {json.dumps(step_payload)}", flush=True)

        if done:
            break

        observation = step_result["observation"]

    grade = env_grade()
    score = grade["score"]
    bd    = grade.get("breakdown", {})

    # [END] log
    end_payload = {
        "run_id": run_id,
        "task_id": task_id,
        "score": score,
        "breakdown": bd,
    }
    print(f"[END] {json.dumps(end_payload)}", flush=True)

    # old score print:
    # print(f"\n  SCORE: {score:.4f} | "
    #       f"cured={bd.get('cured',0)} "
    #       f"failed={bd.get('failed',0)} "
    #       f"overkill={bd.get('overkill',0)} "
    #       f"partial={bd.get('partial',0)}")
    return score


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",   default=None)
    parser.add_argument("--tasks", default="easy,medium,hard")
    args = parser.parse_args()

    if args.url:
        global API_BASE_URL
        API_BASE_URL = args.url.rstrip("/")

    task_list = [t.strip() for t in args.tasks.split(",")]
    scores    = {}

    for task_id in task_list:
        scores[task_id] = run_task(task_id)

    avg = sum(scores.values()) / len(scores)

    # final banner (optional, safe):
    # print(f"\n{'='*60}")
    # print(f"  FINAL SCORES")
    # print(f"{'='*60}")
    # for k, v in scores.items():
        # bar = "=" * int(v * 30)
        # print(f"  {k:8s}: {v:.4f}  [{bar}]")
    # print(f"  {'average':8s}: {avg:.4f}")
    # print(f"{'='*60}\n")

    return avg


if __name__ == "__main__":
    main()
