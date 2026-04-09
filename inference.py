import os, json, argparse, requests, time
from openai import OpenAI
from typing import Optional, List

# ── config ────────────────────────────────────────────────────────────────────
API_BASE_URL   = os.getenv("API_BASE_URL", "https://prakashrajk-antibiotic-stewardship.hf.space")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or os.getenv("HF_TOKEN", "")

client = OpenAI(api_key=OPENAI_API_KEY)

ANTIBIOTIC_NAMES  = {0: "Penicillin", 1: "Azithromycin", 2: "Vancomycin"}
RESISTANCE_DANGER = 0.70
RESISTANCE_WARN   = 0.50

# 503 = backend session not ready (now returned instead of 400)
RETRYABLE_STATUS = {500, 502, 503, 504}  # 500=unhandled server exception


# ── retry helper ──────────────────────────────────────────────────────────────

def _call_with_retry(fn, label: str, retries: int = 10, backoff: float = 10.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            wait = backoff * (attempt + 1)
            print(f"[RETRY] {label}: connection error (attempt {attempt+1}/{retries}), waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
        except requests.exceptions.Timeout as e:
            last_exc = e
            wait = backoff * (attempt + 1)
            print(f"[RETRY] {label}: timeout (attempt {attempt+1}/{retries}), waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = e.response.status_code if e.response is not None else 0
            if status in RETRYABLE_STATUS and attempt < retries - 1:
                wait = backoff * (attempt + 1)
                print(f"[RETRY] {label}: HTTP {status} (attempt {attempt+1}/{retries}), waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
            else:
                print(f"[ERROR] {label}: HTTP {status} — {e}", flush=True)
                raise
    print(f"[ERROR] {label}: all {retries} attempts failed — {last_exc}", flush=True)
    raise last_exc


# ── logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: int, reward: float, done: bool, error=None) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── environment API ───────────────────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    def _do():
        r = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id}, timeout=60)
        r.raise_for_status()
        return r.json()
    return _call_with_retry(_do, label=f"env_reset({task_id})")

def env_step(antibiotic: int, task_id: str = "") -> dict:
    def _do():
        payload = {"antibiotic": antibiotic}
        if task_id:
            payload["task_id"] = task_id          # lets backend route by task_id, not _active_task
        r = requests.post(f"{API_BASE_URL}/step", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    return _call_with_retry(_do, label=f"env_step(action={antibiotic})")

def env_grade(task_id: str = "") -> dict:
    def _do():
        params = {"task_id": task_id} if task_id else {}
        r = requests.get(f"{API_BASE_URL}/grade", params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    return _call_with_retry(_do, label="env_grade()")


# ── deterministic fast-path ───────────────────────────────────────────────────

def deterministic_choice(obs: dict) -> Optional[int]:
    severity   = obs["severity"]
    infection  = obs["infection"]
    age        = obs["age"]
    resistance = {int(k): float(v) for k, v in obs["resistance"].items()}

    def usable(drug_id: int) -> bool:
        return resistance[drug_id] <= RESISTANCE_DANGER

    if infection == "MRSA":
        return 2
    if severity == 3:
        if usable(2): return 2
        if usable(1): return 1
        return 0
    is_vulnerable = age <= 12 or age >= 65
    if is_vulnerable and severity == 1 and usable(0):
        return 0
    if severity == 1 and not usable(0) and not usable(1):
        return 2
    if severity == 2 and not usable(1) and not usable(0):
        return 2
    return None


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
            f"  {emoji} Patient {h['patient_num']}: age={h['age']}, sev={h['severity']}, "
            f"{h['infection']}, gave {h['drug_name']} -> {h['outcome']} ({h['reward']:+.1f})"
        )
    outcomes = [h["outcome"] for h in history]
    lines.append(
        f"\nTotals: {outcomes.count('CURED')} cured, {outcomes.count('FAILED')} failed, "
        f"{outcomes.count('Partial')} partial, {outcomes.count('OVERKILL')} overkill"
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
  severity 1 (mild): prefer Penicillin(0); switch to Azithromycin(1) if resistance[0]>0.50; NEVER use Vancomycin(2)
  severity 2 (moderate): prefer Azithromycin(1); escalate to Vancomycin(2) only if resistance[1]>0.60
  severity 3 (severe): always Vancomycin(2) unless resistance[2]>0.70
  MRSA: always Vancomycin(2)
  Vulnerable (age<=12 or age>=65): use weakest drug that still works
KEY INSIGHT: Stronger drugs unnecessarily burn resistance for ALL future patients.
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
            if antibiotic not in (0, 1, 2):
                raise ValueError(f"Invalid: {antibiotic}")
            return antibiotic
        except Exception as e:
            print(f"[LLM] attempt {attempt+1}/{retries} failed: {e}", flush=True)

    fallback = {1: 0, 2: 1, 3: 2}.get(obs.get("severity", 2), 1)
    print(f"[LLM] using fallback action={fallback}", flush=True)
    return fallback


# ── episode runner ────────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    # ── inner reset guard: if /reset fails after all retries, return 0.0 cleanly
    try:
        reset_data = env_reset(task_id)
    except Exception as e:
        print(f"[run_task] env_reset({task_id}) failed permanently: {e} → scoring 0.0", flush=True)
        return 0.0

    observation = reset_data.get("observation")
    if observation is None:
        print(f"[run_task] /reset returned no observation for {task_id} → scoring 0.0", flush=True)
        return 0.0

    history, rewards = [], []
    step_num = 0

    log_start(task=task_id, env="antibiotic-stewardship", model=MODEL_NAME)

    while True:
        sev, infection, age = observation["severity"], observation["infection"], observation["age"]
        patient_n = observation["patients_treated"] + 1
        step_num  = patient_n

        fast = deterministic_choice(observation)
        antibiotic = fast if fast is not None else ask_llm(observation, history)

        step_result = env_step(antibiotic, task_id=task_id)
        reward  = step_result["reward"]
        done    = step_result["done"]
        info    = step_result.get("info", {})
        outcome = info.get("outcome", "?")

        rewards.append(reward)
        history.append({
            "patient_num": patient_n, "severity": sev, "age": age,
            "infection": infection, "drug_id": antibiotic,
            "drug_name": ANTIBIOTIC_NAMES[antibiotic],
            "outcome": outcome, "reward": reward,
        })

        log_step(step=step_num, action=antibiotic, reward=reward, done=done)

        if done:
            break

        observation = step_result.get("observation")
        if observation is None:
            print(f"[WARN] No observation at step {step_num}, ending early.", flush=True)
            break

    grade   = env_grade(task_id=task_id)
    score   = grade.get("score", 0.0)
    log_end(success=score >= 0.1, steps=step_num, score=score, rewards=rewards)
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
    scores = {}

    for task_id in task_list:
        try:
            scores[task_id] = run_task(task_id)
            print(f"[MAIN] task={task_id} score={scores[task_id]:.3f}", flush=True)
        except Exception as e:
            print(f"[MAIN] task={task_id} FAILED: {e}", flush=True)
            scores[task_id] = 0.0

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"[MAIN] Final average score: {avg:.3f}", flush=True)
    return avg


if __name__ == "__main__":
    main()
