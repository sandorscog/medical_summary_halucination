import pandas as pd


# def read_pharmacy_table():
#     table = pd.read_csv('data_samples/prescription.csv')
#     return table

def context_builder_text(context_jsons: list[dict]) -> str:
    """
    Build a textual medication summary from context JSONs.
    Extracts the 'description' field from each JSON entry and stitches them together.
    """
    medication_descriptions = [
        instance.get("description", "")
        for instance in context_jsons
        if isinstance(instance, dict) and instance.get("description")
    ]

    context = stitch_medicine_descriptions(medication_descriptions)
    return context


def context_builder_json(context: pd.DataFrame) -> list[dict]:
    """
    Build a structured JSON context for medication administration.
    Converts duration to human-readable text (hours or days).
    """
    df = context[['drug', 'prod_strength', 'starttime', 'stoptime']].copy()

    df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
    df['stoptime'] = pd.to_datetime(df['stoptime'], errors='coerce')

    df['administration_length'] = df['stoptime'] - df['starttime']

    context_list = []
    for _, row in df.iterrows():
        if pd.isnull(row['administration_length']):
            duration_text = None
            duration_hours = None
            duration_days = None
        else:
            duration_hours = row['administration_length'].total_seconds() / 3600
            duration_days = duration_hours / 24

            # Decide units (hours or days)
            if duration_hours < 24:
                hrs = int(round(duration_hours))
                unit = "hour" if hrs == 1 else "hours"
                duration_text = f"{hrs} {unit}"
            else:
                days = int(round(duration_days))
                unit = "day" if days == 1 else "days"
                duration_text = f"{days} {unit}"

        entry = {
            "medication_name": row["drug"],
            "strength": row["prod_strength"],
            "start_time": row["starttime"].strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(row["starttime"]) else None,
            "stop_time": row["stoptime"].strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(row["stoptime"]) else None,
            "duration_hours": round(duration_hours, 2) if duration_hours is not None else None,
            "duration_days": round(duration_days, 2) if duration_days is not None else None,
            "description": (
                f"{row['drug']} ({row['prod_strength']}) was administered for {duration_text} "
                if duration_text else None
            )
        }

        context_list.append(entry)

    return context_list


def stitch_medicine_descriptions(descriptions: list[str]) -> str:
    """
    Combine a list of medication descriptions into a single paragraph.
    Handles punctuation, spacing, and empty inputs gracefully.
    """
    header = "The patient received the following medication: "

    # Remove empty or None entries
    descriptions = [d.strip() for d in descriptions if isinstance(d, str) and d.strip()]

    if not descriptions:
        return "No medication was administered."

    if len(descriptions) == 1:
        # Ensure sentence ends with a period
        desc = descriptions[0]
        if not desc.endswith("."):
            desc += "."
        return header + desc

    # Capitalize first letter of each sentence if needed
    descriptions = [d[0].upper() + d[1:] if d and not d[0].isupper() else d for d in descriptions]

    # Join the sentences naturally with spaces
    text = " ".join(descriptions)

    # Ensure proper punctuation at the end
    if not text.endswith("."):
        text += "."

    return header + text





def build_discharge_windows(discharge_df: pd.DataFrame) -> dict:
    """
    Builds and returns a dictionary of discharge windows by subject_id.
    Each entry: subject_id -> list of (hadm_id, window_start, window_end)
    """
    discharge_df = discharge_df.sort_values(['subject_id', 'storetime']).copy()
    discharge_df['window_start'] = discharge_df.groupby('subject_id')['storetime'].shift(1)
    discharge_df['window_start'] = discharge_df['window_start'].fillna(pd.Timestamp.min)
    discharge_df['window_end'] = discharge_df['storetime']

    window_dict = (
        discharge_df
        .groupby('subject_id')
        .apply(lambda df: list(zip(df['hadm_id'], df['window_start'], df['window_end'])))
        .to_dict()
    )

    return window_dict


def get_patient_window_data(subject_id: int,
                            hadm_id: int,
                            windows_dict: dict,
                            patient_df: pd.DataFrame,
                            time_col: str) -> pd.DataFrame:
    """
    Retrieve patient data rows for a specific discharge window.
    """
    if subject_id not in windows_dict:
        return pd.DataFrame(columns=patient_df.columns)

    # Get all windows for this patient
    for w_hadm, start, end in windows_dict[subject_id]:
        if w_hadm == hadm_id:
            mask = (
                (patient_df['subject_id'] == subject_id) &
                (pd.to_datetime(patient_df[time_col]) > start) &
                (pd.to_datetime(patient_df[time_col]) <= end)
            )
            return patient_df.loc[mask].copy()

    return pd.DataFrame(columns=patient_df.columns)

