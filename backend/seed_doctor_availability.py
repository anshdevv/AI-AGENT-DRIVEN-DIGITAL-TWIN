from __future__ import annotations

import json
from pathlib import Path

from backend.config import supabase


SEED_PATH = Path(__file__).resolve().parent / "seeds" / "doctor_availability_seed.json"


def main() -> None:
    if not supabase:
        raise RuntimeError("Supabase is not configured.")

    rows = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    existing = supabase.table("doctor_availability").select("*").execute().data or []
    existing_keys = {
        (
            int(row.get("doctor_id")),
            int(row.get("day_of_week")),
            str(row.get("start_time")),
            str(row.get("end_time")),
        )
        for row in existing
    }

    pending = [
        row
        for row in rows
        if (
            int(row["doctor_id"]),
            int(row["day_of_week"]),
            str(row["start_time"]),
            str(row["end_time"]),
        )
        not in existing_keys
    ]

    if not pending:
        print("doctor_availability already contains the seed rows.")
        return

    response = supabase.table("doctor_availability").insert(pending).execute()
    created = response.data or []
    print(f"Inserted {len(created)} doctor_availability rows.")


if __name__ == "__main__":
    main()
