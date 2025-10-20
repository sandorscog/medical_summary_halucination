import pandas as pd


# def read_pharmacy_table():
#     table = pd.read_csv('data_samples/prescription.csv')
#     return table

def context_builder_text(context) -> str:


    return context


def context_builder_json(context) -> dict:

    df = context[['drug', 'prod_strength', 'starttime', 'stoptime']]

    df['starttime'] = pd.to_datetime(df['starttime'])
    df['stoptime'] = pd.to_datetime(df['stoptime'])

    df['administration_length'] = df['stoptime'] - df['starttime']

    context = []
    for _, row in df.iterrows():
        entry = {
            "medication_name": row["drug"],
            "strength": row["prod_strength"],
            "start_time": row["starttime"].strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(row["starttime"]) else None,
            "stop_time": row["stoptime"].strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(row["stoptime"]) else None,
            "duration_hours": round(row["administration_length"].total_seconds() / 3600, 2)
                if pd.notnull(row["administration_length"]) else None,
            "duration_days": round(row["administration_length"].total_seconds() / (3600 * 24), 2)
                if pd.notnull(row["administration_length"]) else None,
            "description": (
                f"{row['drug']} ({row['prod_strength']}) was administered for "
                f"{round(row['administration_length'].total_seconds() / 3600, 1)} hours "
                f"from {row['starttime'].strftime('%Y-%m-%d %H:%M')} "
                f"to {row['stoptime'].strftime('%Y-%m-%d %H:%M')}."
                if pd.notnull(row['administration_length']) else None
            )
        }
        context.append(entry)

    return context


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

