from backend.config import supabase
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import re

class BookAppointment:
    def __call__(self, state):
        print("======== Reached Book Appointment function ========")
        user_input = state.get("user_input", "").strip()
        print(state)
        
        # --- PHASE 1: REGISTRATION STATE MACHINE ---
        # We determine "who" is booking before we process "what" they are booking.
        IsNewUser = False
        step = state.get("booking_step")
        patient_data = state.get("patient_data", {})
        
        # Default start
        if not step:
            step = "ask_phone"

        # 1. Ask Phone
        if step == "ask_phone":
            # Check if phone is already in input (e.g. "Book for 0300123...")
            phone_match = re.search(r"(\d{10,12})", user_input)
            if phone_match:
                # If found immediately, proceed to check
                user_input = phone_match.group(0) 
                step = "check_phone"
            else:
                state["response"] = "To secure your appointment, please provide your mobile number."
                state["booking_step"] = "check_phone"
                return state

        # 2. Check Phone / Lookup
        if step == "check_phone":
            phone_match = re.search(r"(\d{10,12})", user_input)
            if not phone_match:
                state["response"] = "Please enter a valid 11-digit mobile number."
                return state # Stay here
            
            phone = phone_match.group(0)
            
            # DB Lookup
            print("\nLooking up phone in DB:", phone)
            res = supabase.table("patients").select("*").eq("phone", phone).execute()
            
            if res.data:
                # User Found
                print("\nUser found in DB:", res.data)
                patient = res.data[0]
                print("\nPatient record:", patient)
                state["patient_id"] = patient["id"]
                state["patient_data"] = patient
                # Proceed to Booking Logic
                step = "attempt_booking"
            else:
                # New User
                patient_data["phone"] = phone
                state["patient_data"] = patient_data
                state["response"] = "I don't see an account with this number. What is your Full Name?"
                IsNewUser = True
                state["booking_step"] = "ask_name"
                return state

        # 3. Ask Name
        if step == "ask_name":
            patient_data["name"] = user_input
            state["patient_data"] = patient_data
            state["response"] = "Got it. And your Email Address?"
            state["booking_step"] = "ask_email"
            return state

        # 4. Ask Email & Create
        if step == "ask_email":
            patient_data["email"] = user_input
            
            new_user = {
                "name": patient_data["name"],
                "phone": patient_data["phone"],
                "email": patient_data.get("email"),
            }
            try:
                res = supabase.table("patients").insert(new_user).execute()
                if res.data:
                    state["patient_id"] = res.data[0]["id"]
                    step = "attempt_booking"
                else:
                    state["response"] = "System error creating profile."
                    return state
            except Exception as e:
                print("Error occured after getting email and inserting in patients table")
                state["response"] = f"Database Error: {str(e)}"
                return state

        # --- PHASE 2: YOUR COMPLEX BOOKING LOGIC ---
        # This only runs if step == "attempt_booking" (User is identified)
        
        if step == "attempt_booking":
            PKT = ZoneInfo("Asia/Karachi")
            now = datetime.now(PKT)
            
            # Retrieve extracted entities
            doctor_name = state.get("doctor_name")
            specialization = state.get("specialization")
            user_date_str = state.get("date")
            user_time_str = state.get("time")
            patient_id = state.get("patient_id")

            # --- DATE LOGIC ---
            if not user_date_str:   
                if IsNewUser:
                    state["response"] = f"Thanks {state.get('patient_data', {}).get('name')}. What date would you like to book for?"
                else:
                    state["response"] = f"Thanks {state.get('patient_data', {}).get('Name')}. What date would you like to book for?"
                state["booking_step"] = "attempt_booking" # Stay here
                print("Printing value of state after fetching its name using the phone")
                return state

            # Handle relative dates (Your logic)
            if "today" in user_date_str.lower():
                target_date = now
            elif "tomorrow" in user_date_str.lower():
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

            # --- TIME LOGIC ---
            if not user_time_str:
                state["response"] = "What time would you like? (HH:MM format)"
                return state

            try:
                user_time = datetime.strptime(user_time_str, "%H:%M").time()
            except ValueError:
                state["response"] = "Please provide time in HH:MM (24-hour) format."
                return state

            # --- DOCTOR/SPECIALIZATION FINDING ---
            candidate_doctors = []
            
            # Case A: Doctor Name provided
            if doctor_name:
                res = supabase.table("doctors").select("id, name, specialization").ilike("name", f"%{doctor_name}%").execute()
                candidate_doctors = res.data
                if not candidate_doctors:
                    state["response"] = f"Doctor '{doctor_name}' not found."
                    return state
            
            # Case B: Specialization provided
            elif specialization:
                res = supabase.table("doctors").select("id, name, specialization").ilike("specialization", f"%{specialization}%").execute()
                candidate_doctors = res.data
                if not candidate_doctors:
                    state["response"] = f"No doctors found for {specialization}."
                    return state
            else:
                 state["response"] = "Please specify which doctor or specialization you need."
                 return state

            # --- AVAILABILITY CALCULATION (Your complex loop) ---
            day_map = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}
            day_int = day_map.get(weekday, None)
            chosen_doctor = None
            chosen_slot = None
            
            # Check slots for candidates
            for doc in candidate_doctors:
                avail_res = supabase.table("doctor_availability").select("*").eq("doctor_id", doc["id"]).execute()
                
                for slot in avail_res.data or []:
                    if slot.get("day_of_week") != day_int:
                        continue
                        
                    start_t = datetime.strptime(slot["start_time"], "%H:%M:%S").time()
                    end_t = datetime.strptime(slot["end_time"], "%H:%M:%S").time()

                    if start_t <= user_time < end_t:
                        chosen_doctor = doc
                        chosen_slot = slot
                        break # Found a valid doctor/slot
                
                if chosen_doctor:
                    break # Stop looking at other doctors

            if not chosen_doctor or not chosen_slot:
                state["response"] = f"Sorry, no doctors are available on {weekday} at {user_time_str}."
                return state

            # --- FINAL BOOKING ---
            duration_minutes = int(chosen_slot.get("slot_duration_minutes") or 15)
            start_dt = datetime.strptime(f"{date} {user_time_str}", "%Y/%m/%d %H:%M")
            end_dt = start_dt + timedelta(minutes=duration_minutes)
            slot_payload = {
                "doctor_id": chosen_doctor["id"],
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "status": "booked",
            }
            slot_res = supabase.table("slots").insert(slot_payload).execute()
            slot_id = slot_res.data[0]["id"] if slot_res.data else None
            if not slot_id:
                state["response"] = "Could not reserve a time slot for that appointment. Please try a different time."
                return state

            appointment = {
                "patient_id": patient_id,
                "doctor_id": chosen_doctor["id"],
                "slot_id": slot_id,
                "status": "booked",
            }
            try:
                appointment_res = supabase.table("appointments").insert(appointment).execute()
                appointment_id = appointment_res.data[0]["id"] if appointment_res.data else None
                if appointment_id:
                    supabase.table("appointment_events").insert(
                        {"appointment_id": appointment_id, "event_type": "created"}
                    ).execute()
                
                state["response"] = (
                    f"✅ Appointment Confirmed!\n"
                    f"Doctor: Dr. {chosen_doctor['name']}\n"
                    f"Date: {date} ({weekday})\n"
                    f"Time: {user_time_str}\n\n"
                    "I just need to ask a few quick medical questions to prepare the doctor."
                )
                
                # --- PHASE 3: TRIGGER TRIAGE ---
                state["booking_step"] = "done"
                state["triage_active"] = True
                
                raw_complaint = state.get("patient_complaint", "").lower()
            
                # 2. Logic to pick the correct Protocol File (.md)
                if raw_complaint:
                    # Map specific words to file names
                    if "cough" in raw_complaint or "cold" in raw_complaint or "throat" in raw_complaint:
                        state["triage_symptom"] = "cough"
                    elif "chest" in raw_complaint or "heart" in raw_complaint:
                        state["triage_symptom"] = "chest_pain"
                    elif "stomach" in raw_complaint or "belly" in raw_complaint or "abdominal" in raw_complaint:
                        state["triage_symptom"] = "stomach_pain"
                    elif "head" in raw_complaint or "dizzy" in raw_complaint:
                        state["triage_symptom"] = "headache"
                    elif "rash" in raw_complaint or "itch" in raw_complaint or "skin" in raw_complaint:
                        state["triage_symptom"] = "rash"
                    elif "fever" in raw_complaint or "temperature" in raw_complaint:
                        state["triage_symptom"] = "fever"
                    else:
                        # We have a complaint (e.g. "leg pain") but no specific file.
                        state["triage_symptom"] = "general"
                
                else:
                    # 3. Fallback: User didn't state a symptom (e.g. just said "Book Dr. Ali")
                    # Infer category from the Doctor's Specialization
                    spec = chosen_doctor.get("Specialization", "").lower()
                    
                    if "physician" in spec: state["triage_symptom"] = "fever" # Generic default
                    elif "cardiologist" in spec: state["triage_symptom"] = "chest_pain"
                    elif "dermatologist" in spec: state["triage_symptom"] = "rash"
                    elif "neurologist" in spec: state["triage_symptom"] = "headache"
                    else: state["triage_symptom"] = "general"

            # Note: We keep 'patient_complaint' in the state so Triage.py can read it
            except Exception as e:
                state["response"] = f"Booking Failed: {str(e)}"
            
            return state

        return state