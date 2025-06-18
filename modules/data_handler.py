import pandas as pd

def load_discharges():
    data = pd.read_csv('discharges.csv') ### TODO: Define the file name/path
    return data

def load_patient_data():
    data = pd.read_csv('patients.csv') ### TODO: Define the file name/path
    return data

def load_data():
    discharges = load_discharges()
    patient_data = load_patient_data()
    merged_data = pd.merge(discharges, patient_data, on='subject_id', how='left')
    return merged_data

