import os

import pandas as pd

def load_discharges():
    data = pd.read_csv('../data_samples/notes/discharge.csv')
    return data[:5] ### TODO: Remove limit in the return

def load_patient_data():

    admissions = pd.read_csv('../data_samples/hosp/admissions.csv')
    lab_events = pd.read_csv('../data_samples/hosp/labevents.csv')
    prescriptions = pd.read_csv('../data_samples/hosp/prescriptions.csv')
    pharmacy = pd.read_csv('../data_samples/hosp/pharmacy.csv')
    emar = pd.read_csv('../data_samples/hosp/emar.csv')
    omr = pd.read_csv('../data_samples/hosp/omr.csv')
    transfers = pd.read_csv('../data_samples/hosp/transfers.csv')

    


    return data

def load_data():
    discharges = load_discharges()
    patient_data = load_patient_data()
    merged_data = pd.merge(discharges, patient_data, on='subject_id', how='left')
    return merged_data

