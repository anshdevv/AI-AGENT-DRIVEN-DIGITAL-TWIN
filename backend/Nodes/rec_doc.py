from backend.config import supabase
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

class RecommendDoctor:
    def __call__(self, state):

        user_input = state.get("user_input", "").lower()
        PKT = ZoneInfo("Asia/Karachi")

        specialization = state.get("specialization")
        user_date_str = state.get("date")
        doctor_name = state.get("doctor_name")
        user_time_str = state.get("time")  # expected "HH:MM"
        print(state)

        now = datetime.now(PKT)

        if not user_date_str and doctor_name:
            doc_res = (
                supabase.table("Doctors")
                .select("Name, doctor_availability(days, start_time, end_time)")  # fixed syntax
                .ilike("Name", f"%{doctor_name}%")
                .execute()
            )

            if not doc_res.data:
                state["response"] = f"No availability found for Dr. {doctor_name}."
                return state

            doc = doc_res.data[0]  # first doctor
            avail = doc.get("doctor_availability", [])
            
            if avail:
                slot = avail[0]  # first available slot
                response = (
                    f"{doc['Name']} is available on {slot['days']} "
                    f"from {slot['start_time']} to {slot['end_time']}"
                )
            else:
                response = f"No availability found for Dr. {doctor_name}."

            state["response"] = response
            return state

        
        if not user_date_str:
            state["response"] = "Please provide a valid date."
            return state

        # --- Handle relative dates ---
        if "today" in user_date_str.lower():
            print("inside today")
            target_date = now
            print(target_date)
        elif "tomorrow" in user_date_str.lower():
            target_date = now + timedelta(days=1)
        elif "day after tomorrow" in user_date_str.lower():
            target_date = now + timedelta(days=2)
        else:
            try:
                target_date = datetime.strptime(user_date_str, "%Y/%m/%d")
                print(target_date)
            except ValueError:
                state["response"] = "Please provide a valid date in YYYY/MM/DD format."
                return state

        date = target_date.strftime("%Y/%m/%d")
        weekday = target_date.strftime("%a").lower()  # mon, tue, wed...

        if not specialization:
            state["response"] = "Please specify the specialization you are looking for."
            return state

        try:
            # === Step 1: Get doctors by specialization ===
            doctors_res = (
                supabase.table("Doctors")
                .select("*")
                .ilike("Specialization", f"%{specialization}%")
                .execute()
            )

            if not doctors_res.data:
                state["response"] = f"Sorry, I couldn’t find any available {specialization}s right now."
                return state

            day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

            def is_day_in_range(day_range, target):
                """Return True if target day (e.g. 'wed') falls in a textual range like 'tue-thu'."""
                parts = [p.strip().lower() for p in day_range.split('-')]
                if len(parts) == 1:
                    return parts[0] == target
                start, end = parts
                s, e, t = day_order.index(start), day_order.index(end), day_order.index(target)
                if s <= e:
                    return s <= t <= e
                else:
                    # wrap-around e.g. sat-mon
                    return t >= s or t <= e

            available_doctors = []

            # Convert user time string → datetime.time
            user_time = None
            if user_time_str:
                try:
                    user_time = datetime.strptime(user_time_str, "%H:%M").time()
                except ValueError:
                    state["response"] = "Please provide time in HH:MM (24-hour) format."
                    return state

            for doc in doctors_res.data:
                avail_res = (
                    supabase.table("doctor_availability")
                    .select("*")
                    .eq("doctor_id", doc["id"])
                    .execute()
                )

                for slot in avail_res.data:
                    days_text = slot["days"].strip().lower()

                    if not is_day_in_range(days_text, weekday):
                        continue

                    if user_time:
                        # Convert Supabase TIME strings to Python time objects
                        start_t = datetime.strptime(slot["start_time"], "%H:%M:%S").time()
                        end_t = datetime.strptime(slot["end_time"], "%H:%M:%S").time()
                        if not (start_t <= user_time <= end_t):
                            continue

                    available_doctors.append(doc)
                    break  # found one matching slot, no need to check others

            if not available_doctors:
                state["response"] = (
                    f"No {specialization} doctors are available"
                    + (f" on {date} at {user_time_str}" if user_time_str else "")
                    + "."
                )
                return state

            # === Step 3: Prepare response ===
            doctor_list = "\n".join([
                f"- Dr. {d['Name']} ({d.get('Qualification', 'N/A')}, {d.get('Experience', 0)} yrs exp, Room {d.get('room', 'N/A')})"
                for d in available_doctors
            ])

            state["response"] = (
                f"I recommend consulting a {specialization}.\n"
                f"Here are some available doctors"
                + (f" on {date} at {user_time_str}" if user_time_str else "")
                + f":\n{doctor_list}\n"
                f"Would you like to book an appointment with one?"
            )

        except Exception as e:
            state["response"] = f"Error fetching doctor data: {e}"

        return state
