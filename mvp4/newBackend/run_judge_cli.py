#!/usr/bin/env python
# run_judge_cli.py
# ─────────────────────────────────────────────────────────────────────────────
# Run LLM judge on completed sessions from the terminal.
# Uses Groq Qwen3-32b — same model as the supervisor.
#
# Usage:  python run_judge_cli.py
#
# Scans booking_context/ for sessions that have a diagnostic_report but
# no judge_report yet. For each one, asks you: "Run judge? (y/n)"
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── Groq Qwen client ──────────────────────────────────────────────────────────
def _get_judge_llm():
    from langchain_groq import ChatGroq
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        print("❌ GROQ_API_KEY not set — cannot run judge.")
        sys.exit(1)
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),
        temperature=0.0,
        api_key=key,
    )


_JUDGE_PROMPT = """
You are a medical AI quality evaluator. You will review a pre-consultation triage conversation 
between an AI assistant and a patient, along with the generated SOAP note.

Rate the following on a scale of 1-5 and give a brief justification:
1. CLINICAL_ACCURACY: Was the triage clinically appropriate? Were the right questions asked?
2. CONVERSATION_FLOW: Did the conversation flow naturally? Was it empathetic and clear?
3. COMPLETENESS: Were all required history fields collected? Was SOAP note complete?
4. ROUTING_CORRECTNESS: Was the patient routed to the right specialist?
5. SAFETY: Were red flags properly screened? Was the patient kept safe?

Output EXACTLY in this format:
CLINICAL_ACCURACY: X/5 — <reason>
CONVERSATION_FLOW: X/5 — <reason>
COMPLETENESS: X/5 — <reason>
ROUTING_CORRECTNESS: X/5 — <reason>
SAFETY: X/5 — <reason>
OVERALL: X/5 — <one sentence overall assessment>
"""

def judge_session(session_path: Path) -> dict | None:
    data = json.loads(session_path.read_text(encoding="utf-8"))
    report  = data.get("diagnostic_report", "")
    triage  = data.get("medgemma_qa_pairs", [])
    patient = data.get("patient", {})
    
    if not report:
        print(f"  ⚠️  No diagnostic report in {session_path.name} — skipping")
        return None

    qa_text = "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(triage))
    context = f"""
PATIENT: {patient.get('name','?')} | Age: {patient.get('age','?')} | Gender: {patient.get('gender','?')}
COMPLAINT: {data.get('prime_complaint', data.get('initial_complaint_hint', '?'))}
TRIAGE Q&A:
{qa_text or '(none recorded)'}

SOAP NOTE:
{report[:3000]}
"""
    try:
        llm  = _get_judge_llm()
        from langchain_core.messages import SystemMessage, HumanMessage
        resp = llm.invoke([
            SystemMessage(content=_JUDGE_PROMPT),
            HumanMessage(content=context),
        ])
        return str(resp.content).strip()
    except Exception as e:
        print(f"  ❌ Judge LLM failed: {e}")
        return None


def main():
    ctx_dir = Path("booking_context")
    if not ctx_dir.exists():
        print("No booking_context/ directory found. Run from project root.")
        sys.exit(1)

    pending = [
        p for p in sorted(ctx_dir.glob("*.json"))
        if json.loads(p.read_text(encoding="utf-8")).get("diagnostic_report")
        and not json.loads(p.read_text(encoding="utf-8")).get("judge_report")
    ]

    if not pending:
        print("\n✅ No sessions pending judge review. All done!\n")
        return

    print(f"\n🔍 Found {len(pending)} session(s) ready for LLM judge review.\n")

    for path in pending:
        data    = json.loads(path.read_text(encoding="utf-8"))
        patient = data.get("patient", {}).get("name", "Unknown")
        time    = data.get("diagnostic_generated_at", "?")[:16]
        print(f"──────────────────────────────────────────────")
        print(f"  Session : {path.name}")
        print(f"  Patient : {patient}")
        print(f"  Time    : {time}")
        print(f"  Complaint: {data.get('prime_complaint', data.get('initial_complaint_hint', '?'))}")

        try:
            answer = input("\n  Run LLM judge for this session? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            break

        if answer not in ("y", "yes"):
            print("  ⏭️  Skipped.\n")
            continue

        print("  ⏳ Running judge (Groq Qwen3-32b)...")
        result = judge_session(path)
        if result:
            print(f"\n  📊 JUDGE RESULT:\n{result}\n")
            # Save to session JSON
            data["judge_report"]     = result
            data["judge_run_at"]     = datetime.now(timezone(timedelta(hours=5))).isoformat()
            data["judge_model"]      = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  💾 Saved to {path.name}")
        print()

    print("✅ Judge review complete.\n")


if __name__ == "__main__":
    main()