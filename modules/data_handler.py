import os

import pandas as pd

def load_discharges():
    data = pd.read_csv('data_samples/notes/discharge.csv')
    return data[:3]

def load_patient_data():
    data = pd.read_csv('data_samples/hosp/prescriptions.csv') ### TODO: Remove limit in the return
    return data[:100] 

def load_data():
    discharges = load_discharges()
    patient_data = load_patient_data()
    merged_data = pd.merge(discharges, patient_data, on='subject_id', how='left')
    return merged_data

