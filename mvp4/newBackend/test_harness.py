#!/usr/bin/env python3
"""
End-to-End Test Harness — Simulator Edition
═══════════════════════════════════════════
Runs simulated patient conversations against the live orchestrator.
Replaces the old static `turns` list with a dynamic patient simulator that:
  - Reads the bot's reply
  - Answers from a persona JSON via keyword router (no LLM, fast)
  - Falls back to local Ollama 4B model for open-ended replies

Usage:
  python test_harness.py                          # run all personas/*.json
  python test_harness.py --start 0 --end 3        # subset
  python test_harness.py --personas-dir personas  # custom dir
  python test_harness.py --max-turns 50           # safety cap
"""

import argparse
import csv
import json
import os
import re
import time
import sys
import traceback
from datetime import datetime
from pathlib import Path

import requests

from patient_simulator import (
    simulate_patient_reply,
    is_done,
    load_personas,
)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_BACKEND   = "http://localhost:8000"
DEFAULT_PERSONAS  = Path("personas")
RESULTS_DIR       = Path("research_results")
TIMEOUT_SEC       = 300    # per /chat call (Ollama-backed bot can be slow)
TURN_DELAY_SEC    = 0.5
CONV_DELAY_SEC    = 3
DEFAULT_MAX_TURNS = 40     # safety cap — kill any runaway conversation

# ── ANSI colors ───────────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"; RED    = "\033[91m"; YELLOW = "\033[93m"
    BLUE   = "\033[94m"; CYAN   = "\033[96m"; GRAY   = "\033[90m"
    MAGENTA= "\033[95m"; BOLD   = "\033[1m" ; RESET = "\033[0m"

def color(text, c): return f"{c}{text}{C.RESET}"


# ── Bot interaction ───────────────────────────────────────────────────────────
def send_message(backend: str, session_id: str, user_msg: str, log_file) -> dict:
    """POST one user message to /chat, return bot reply dict."""
    payload = {
        "session_id" : session_id,
        "user_input" : user_msg,
        "channel"    : "harness",
        "language"   : "en",
    }
    try:
        r = requests.post(
            f"{backend}/chat",
            json    = payload,
            timeout = TIMEOUT_SEC,
            headers = {"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        log_file.write(f"\n[YOU] {user_msg}\n")
        log_file.write(f"[BOT] {data.get('reply', '')}\n")
        log_file.flush()
        return {"ok": True, "reply": data.get("reply", ""), "raw": data}
    except requests.Timeout:
        log_file.write(f"\n[YOU] {user_msg}\n[ERROR] timeout after {TIMEOUT_SEC}s\n")
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        log_file.write(f"\n[YOU] {user_msg}\n[ERROR] {e}\n")
        return {"ok": False, "error": str(e)}


def run_conversation(backend: str, persona: dict, log_path: Path,
                     max_turns: int) -> dict:
    """Walk through one full simulated conversation. Returns summary dict."""
    session_id = persona["session_id"]
    summary = {
        "id"           : persona["id"],
        "scenario"     : persona.get("scenario", ""),
        "patient"      : persona.get("name", ""),
        "phone"        : persona.get("phone", ""),
        "complaint"    : persona.get("complaint", {}).get("main", ""),
        "expected"     : persona.get("expected_doctor", ""),
        "turns_total"  : 0,
        "turns_persona": 0,
        "turns_llm"    : 0,
        "turns_failed" : 0,
        "status"       : "pending",
        "duration_s"   : 0,
        "ended_reason" : "",
    }
    start = time.time()

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"╔══════════════════════════════════════════════╗\n")
        log_file.write(f"║  Case: {persona['id']:<37} ║\n")
        log_file.write(f"║  Scenario: {persona.get('scenario','')[:32]:<32}  ║\n")
        log_file.write(f"╚══════════════════════════════════════════════╝\n")
        log_file.write(f"Session ID: {session_id}\n")
        log_file.write(f"Phone:      {persona.get('phone','')}\n")
        log_file.write(f"Started:    {datetime.now().isoformat()}\n")

        # First message — the patient initiates with a greeting
        user_msg = "Hello"

        for turn in range(1, max_turns + 1):
            print(color(f"  [{turn:>2}] >>> ", C.GRAY) +
                  color(user_msg[:80], C.YELLOW))

            result = send_message(backend, session_id, user_msg, log_file)
            summary["turns_total"] += 1

            if not result["ok"]:
                print(color(f"        ❌ {result['error']}", C.RED))
                summary["turns_failed"] += 1
                summary["status"]       = "failed_at_turn_" + str(turn)
                summary["ended_reason"] = f"backend_error: {result['error']}"
                log_file.write(f"\n❌ ABORTED at turn {turn}: {result['error']}\n")
                return _finalize(summary, start)

            bot_reply = result["reply"]
            print(color(f"        <<< {bot_reply[:90]}{'...' if len(bot_reply) > 90 else ''}",
                        C.CYAN))

            # Check for natural conversation end
            if is_done(bot_reply):
                summary["status"]       = "complete"
                summary["ended_reason"] = "bot_finished"
                log_file.write(f"\n✅ Bot signalled end of conversation at turn {turn}.\n")
                return _finalize(summary, start)

            # Generate next patient reply
            user_msg, source = simulate_patient_reply(persona, bot_reply)
            if source == "persona":
                summary["turns_persona"] += 1
                src_tag = color("[persona]", C.GREEN)
            else:
                summary["turns_llm"] += 1
                src_tag = color("[llm    ]", C.MAGENTA)
            print(color(f"        {src_tag} next reply queued", C.GRAY))

            log_file.write(f"  (source: {source})\n")
            time.sleep(TURN_DELAY_SEC)

        summary["status"]       = "max_turns"
        summary["ended_reason"] = f"hit max_turns={max_turns}"
        log_file.write(f"\n⚠️  Hit max_turns cap ({max_turns}).\n")
        return _finalize(summary, start)


def _finalize(summary, start):
    summary["duration_s"] = round(time.time() - start, 1)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=None,
                        help="Index (exclusive). Default: run all personas")
    parser.add_argument("--personas-dir", default=str(DEFAULT_PERSONAS),
                        help="Directory containing per-case persona JSONs")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                        help=f"Safety cap per conversation (default {DEFAULT_MAX_TURNS})")
    args = parser.parse_args()

    personas_dir = Path(args.personas_dir)
    if not personas_dir.is_dir():
        print(color(f"❌ {personas_dir} not found.", C.RED))
        sys.exit(1)

    personas = load_personas(personas_dir)
    if not personas:
        print(color(f"❌ No persona JSONs in {personas_dir}/", C.RED))
        sys.exit(1)

    end = args.end if args.end is not None else len(personas)
    subset = personas[args.start:end]

    # Setup output dirs
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Backend health check
    try:
        requests.get(args.backend + "/", timeout=5)
    except Exception as e:
        print(color(f"❌ Backend not reachable at {args.backend}: {e}", C.RED))
        print(color("   Run: uvicorn main:app --host 0.0.0.0 --port 8000", C.GRAY))
        sys.exit(1)

    # Ollama health check
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    ollama_base = ollama_url.rsplit("/api/", 1)[0] if "/api/" in ollama_url else ollama_url
    try:
        requests.get(ollama_base + "/api/tags", timeout=3)
        print(color(f"✓ Ollama reachable at {ollama_base}", C.GREEN))
    except Exception:
        print(color(f"⚠️  Ollama not reachable at {ollama_base} — LLM fallback will fail.", C.YELLOW))
        print(color("   Run: ollama serve   (in a separate terminal)", C.GRAY))

    print(color("\n╔══════════════════════════════════════════════════════╗", C.BOLD))
    print(color(f"║  Simulator harness — {len(subset)} personas                  ║", C.BOLD))
    print(color(f"║  Backend: {args.backend:<40} ║", C.BOLD))
    print(color(f"║  Output:  {str(run_dir):<40} ║", C.BOLD))
    print(color("╚══════════════════════════════════════════════════════╝\n", C.BOLD))

    summaries = []
    for idx, persona in enumerate(subset):
        global_idx = args.start + idx
        print(color(f"\n━━━ [{global_idx + 1}/{end}] {persona['id']} — "
                    f"{persona.get('scenario', '')} ━━━", C.GREEN + C.BOLD))

        log_path = run_dir / f"{persona['id']}.log"
        try:
            summary = run_conversation(args.backend, persona, log_path,
                                       args.max_turns)
        except KeyboardInterrupt:
            print(color("\n\n⏹  Interrupted by user.", C.YELLOW))
            break
        except Exception as e:
            print(color(f"❌ Conversation crashed: {e}", C.RED))
            traceback.print_exc()
            summary = {
                "id"        : persona["id"],
                "scenario"  : persona.get("scenario", ""),
                "status"    : "crashed",
                "error"     : str(e),
                "duration_s": 0,
            }

        summaries.append(summary)

        # Write summary JSON every iteration (in case of crash)
        (run_dir / "_summary.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        status_color = C.GREEN if summary.get("status") == "complete" else C.YELLOW
        print(color(
            f"  → {summary.get('status')} ({summary.get('duration_s')}s, "
            f"persona={summary.get('turns_persona',0)} llm={summary.get('turns_llm',0)}, "
            f"ended: {summary.get('ended_reason','?')})",
            status_color))

        if idx < len(subset) - 1:
            time.sleep(CONV_DELAY_SEC)

    # CSV summary
    csv_path = run_dir / "_summary.csv"
    if summaries:
        keys = ["id","patient","phone","complaint","scenario","expected",
                "status","turns_total","turns_persona","turns_llm","turns_failed",
                "duration_s","ended_reason"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)

    ok      = sum(1 for s in summaries if s.get("status") == "complete")
    failed  = len(summaries) - ok
    total_s = sum(s.get("duration_s", 0) for s in summaries)
    llm_total     = sum(s.get("turns_llm", 0) for s in summaries)
    persona_total = sum(s.get("turns_persona", 0) for s in summaries)
    total_turns   = llm_total + persona_total

    print(color("\n╔══════════════════════════════════════════════════════╗", C.BOLD))
    print(color(f"║  FINAL: {ok}/{len(summaries)} complete   {failed} other states         ", C.BOLD))
    print(color(f"║  Time:  {round(total_s/60, 1)} min                                ", C.BOLD))
    if total_turns:
        print(color(f"║  Router coverage: {persona_total}/{total_turns} turns "
                    f"({100*persona_total//total_turns}%)        ", C.BOLD))
    print(color(f"║  Output dir: {str(run_dir):<40}", C.BOLD))
    print(color("╚══════════════════════════════════════════════════════╝\n", C.BOLD))


if __name__ == "__main__":
    main()