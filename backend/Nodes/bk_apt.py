import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from backend.config import supabase, GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI

# === Initialize ===
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
PKT = ZoneInfo("Asia/Karachi")

class BookAppointment:
    def __call__(self, state):

        user_input = state.get("user_input", "")
        patient_id = state.get("patient_id", 1)

        # data = json.loads(clean_json.group())

        doctor_name = state.get("doctor_name")
        specialization = state.get("specialization")
        user_date_str = state.get("date")
        user_time_str = state.get("time")
        print(state)

        print(doctor_name, specialization, user_date_str, user_time_str)

        now = datetime.now(PKT)

        if not user_date_str:
            state["response"] = "Please provide a valid date."
            return state

        # --- Step 2: Handle relative/explicit dates ---
                # --- Handle relative dates ---
        if "today" in user_date_str.lower():
            print("inside today")
            target_date = now
            print(target_date)
            
        if "tomorrow" in user_date_str.lower():
            target_date = now + timedelta(days=1)
        elif "day after tomorrow" in user_date_str.lower():
            target_date = now + timedelta(days=2)
        else:
            try:
                target_date = datetime.strptime(user_date_str, "%Y/%m/%d")
            except ValueError:
                state["response"] = "Please provide a valid date in YYYY/MM/DD format."
                return state

        date = target_date.strftime("%Y/%m/%d")
        weekday = target_date.strftime("%a").lower()

        if not user_time_str:
            state["response"] = "Please provide a valid time in HH:MM (24-hour) format."
            return state

        # Convert time string to Python time object
        try:
            user_time = datetime.strptime(user_time_str, "%H:%M").time()
        except ValueError:
            state["response"] = "Please provide time in HH:MM (24-hour) format."
            return state

        # --- Step 3: If doctor not mentioned, find by specialization ---
        if not doctor_name:
            if not specialization:
                state["response"] = "Please mention the doctor or your health concern (e.g., bones, heart, skin)."
                return state

            doctors_res = (
                supabase.table("Doctors")
                .select("*")
                .ilike("Specialization", f"%{specialization}%")
                .execute()
            )

            if not doctors_res.data:
                state["response"] = f"No doctors found for specialization '{specialization}'."
                return state

            # === Day range handling logic ===
            day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

            def is_day_in_range(day_range, target):
                parts = [p.strip().lower() for p in day_range.split('-')]
                if len(parts) == 1:
                    return parts[0] == target
                start, end = parts
                s, e, t = day_order.index(start), day_order.index(end), day_order.index(target)
                if s <= e:
                    return s <= t <= e
                else:
                    return t >= s or t <= e  # wrap-around case

            available_doctors = []

            for doc in doctors_res.data:
                avail_res = (
                    supabase.table("doctor_availability")
                    .select("*")
                    .eq("doctor_id", doc["id"])
                    .execute()
                )

                for slot in avail_res.data:
                    if not is_day_in_range(slot["days"], weekday):
                        continue

                    start_t = datetime.strptime(slot["start_time"], "%H:%M:%S").time()
                    end_t = datetime.strptime(slot["end_time"], "%H:%M:%S").time()

                    if start_t <= user_time <= end_t:
                        available_doctors.append(doc)
                        break

            if not available_doctors:
                state["response"] = f"No {specialization} doctors are available on {date} at {user_time_str}."
                return state

            doctor_list = "\n".join([
                f"- Dr. {d['Name']} ({d.get('Experience', 0)} yrs exp)"
                for d in available_doctors
            ])

            state["response"] = (
                f"Available {specialization} doctors on {date} at {user_time_str}:\n"
                f"{doctor_list}\n\n"
                "Please tell me which doctor you’d like to book an appointment with."
            )
            return state

        # --- Step 4: Doctor mentioned: lookup directly ---
        doctor_res = (
            supabase.table("Doctors")
            .select("id, Name")
            .ilike("Name", f"%{doctor_name}%")
            .execute()
        )
        print(doctor_res.data)

        if not doctor_res.data:
            state["response"] = f"Doctor '{doctor_name}' not found."
            return state

        doctor = doctor_res.data[0]
        doctor_id = doctor["id"]

        # === Step 5: Check availability (with range + time handling) ===
        avail_res = (
            supabase.table("doctor_availability")
            .select("*")
            .eq("doctor_id", doctor_id)
            .execute()
        )

        day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

        def is_day_in_range(day_range, target):
            parts = [p.strip().lower() for p in day_range.split('-')]
            if len(parts) == 1:
                return parts[0] == target
            start, end = parts
            s, e, t = day_order.index(start), day_order.index(end), day_order.index(target)
            if s <= e:
                return s <= t <= e
            else:
                return t >= s or t <= e

        available = False
        for slot in avail_res.data:
            if not is_day_in_range(slot["days"], weekday):
                continue

            start_t = datetime.strptime(slot["start_time"], "%H:%M:%S").time()
            end_t = datetime.strptime(slot["end_time"], "%H:%M:%S").time()

            if start_t <= user_time <= end_t:
                available = True
                break

        if not available:
            state["response"] = f"{doctor['Name']} is not available on {weekday} at {user_time_str}."
            return state

        # --- Step 6: Book appointment ---
        appointment = {
            "patient_id": patient_id,
            "appointment_date": date,
            "time": user_time_str,
            "doctor_id": doctor_id,
        }
        supabase.table("appointments").insert(appointment).execute()

        state["response"] = (
            f"✅ Appointment booked successfully!\n"
            f"Doctor: {doctor['Name']}\n"
            f"Date: {date}\n"
            f"Time: {user_time_str} (PKT)"
        )

        return state
