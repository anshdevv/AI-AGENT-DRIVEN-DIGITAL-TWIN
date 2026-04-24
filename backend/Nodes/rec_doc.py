from backend.config import supabase
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class RecommendDoctor:
    def __call__(self, state):
        # --- 1. SETUP & INPUTS ---
        user_input = state.get("user_input", "").lower()
        PKT = ZoneInfo("Asia/Karachi")
        now = datetime.now(PKT)
        print(state)

        specialization = state.get("specialization")
        doctor_name = state.get("doctor_name")
        
        # Date and Time inputs
        user_date_str = state.get("date")
        user_time_str = state.get("time")  # expected "HH:MM"

        print(f"State: {state}")

        # --- 2. EMPATHY LOGIC ---
        # If we inferred a specialization from symptoms but user didn't ask for a specific doctor
        intro_text = ""
        if specialization and not doctor_name:
             intro_text = f"Oh, I'm sorry to hear that. For symptoms like {user_input}, you should consult a {specialization}."

        # --- 3. INTELLIGENT DATE PARSING ---
        # FIX: Do not stop if date is missing. Default to TODAY.
        target_date = now # Default
        date_source = "today" # For the final message

        if user_date_str:
            if "today" in user_date_str.lower():
                target_date = now
                date_source = "today"
            elif "tomorrow" in user_date_str.lower():
                target_date = now + timedelta(days=1)
                date_source = "tomorrow"
            elif "day after tomorrow" in user_date_str.lower():
                target_date = now + timedelta(days=2)
                date_source = user_date_str
            else:
                try:
                    target_date = datetime.strptime(user_date_str, "%Y/%m/%d")
                    date_source = user_date_str
                except ValueError:
                    state["response"] = "Please provide a valid date in YYYY/MM/DD format."
                    return state

        # Calculate database lookup values
        date_display = target_date.strftime("%Y/%m/%d")
        weekday = target_date.strftime("%a").lower()  # mon, tue, wed...

        # --- 4. FETCH CANDIDATE DOCTORS ---
        candidate_doctors = []

        try:
            if doctor_name:
                # Scenario A: User asked for a specific doctor
                res = (supabase.table("doctors")
                       .select("*")
                       .ilike("name", f"%{doctor_name}%")
                       .execute())
                candidate_doctors = res.data
                if not candidate_doctors:
                    state["response"] = f"Doctor '{doctor_name}' not found."
                    return state

            elif specialization:
                # Scenario B: User has symptoms/specialization
                res = (supabase.table("doctors")
                       .select("*")
                       .ilike("specialization", f"%{specialization}%")
                       .execute())
                candidate_doctors = res.data
                if not candidate_doctors:
                    prefix = intro_text + " " if intro_text else ""
                    state["response"] = f"{prefix}Sorry, I couldn’t find any {specialization}s available."
                    return state
            else:
                # Scenario C: No name, no specialization
                state["response"] = "Please specify a doctor name or describe your symptoms."
                return state

            day_map = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}
            day_int = day_map.get(weekday)

            # Parse user time if provided
            user_time_obj = None
            if user_time_str:
                try:
                    user_time_obj = datetime.strptime(user_time_str, "%H:%M").time()
                except ValueError:
                    state["response"] = "Please provide time in HH:MM (24-hour) format."
                    return state

            available_doctors = []

            for doc in candidate_doctors:
                # Fetch slots for this doctor
                slots = (supabase.table("doctor_availability")
                         .select("*")
                         .eq("doctor_id", doc["id"])
                         .execute()).data or []
                
                doc_is_free = False
                valid_slot_str = ""

                for slot in slots:
                    if slot.get("day_of_week") != day_int:
                        continue

                    start_t = datetime.strptime(slot["start_time"], "%H:%M:%S").time()
                    end_t = datetime.strptime(slot["end_time"], "%H:%M:%S").time()
                    
                    if user_time_obj:
                        if start_t <= user_time_obj < end_t:
                            doc_is_free = True
                            valid_slot_str = f"{start_t.strftime('%H:%M')} - {end_t.strftime('%H:%M')}"
                            break
                    else:
                        # If no specific time requested, they are available if they work today
                        doc_is_free = True
                        valid_slot_str = f"{start_t.strftime('%H:%M')} - {end_t.strftime('%H:%M')}"
                        break # Found a valid slot for today
                
                if doc_is_free:
                    # Append doctor with the specific slot info for display
                    doc['display_slot'] = valid_slot_str
                    available_doctors.append(doc)

            # --- 6. CONSTRUCT RESPONSE ---
            if not available_doctors:
                prefix = intro_text + " " if intro_text else ""
                time_msg = f" at {user_time_str}" if user_time_str else ""
                state["response"] = f"{prefix}No doctors found available on {date_source} ({weekday}){time_msg}."
                return state

            # Create the list
            list_text = ""
            for d in available_doctors:
                list_text += f"- Dr. {d['name']} ({d.get('experience_years', 0)} yrs exp) | Time: {d['display_slot']} | Day: {weekday}\n"

            header = intro_text if intro_text else f"Here are the doctors available for {date_source} ({weekday}):"
            
            state["response"] = (
                f"{header}\n\n"
                f"{list_text}\n"
                f"Would you like to book an appointment with any of them?"
            )
            
            # Automatically fill date in state for the next step (Booking)
            state["date"] = date_display

        except Exception as e:
            state["response"] = f"System Error: {str(e)}"

        return state