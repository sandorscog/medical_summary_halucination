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

    # 1. Dimension tables for human-readable descriptions
    d_labitems = pd.read_csv('../data_samples/hosp/d_labitems.csv')
    d_icd_diag = pd.read_csv('../data_samples/hosp/d_icd_diagnoses.csv')

    # 2. The "Medical Conclusion"
    diagnoses_icd = pd.read_csv('../data_samples/hosp/diagnoses_icd.csv')

    # 3. Clinical Specialty
    services = pd.read_csv('../data_samples/hosp/services.csv')
    
    return {

    }


def load_clinical_context():

    NUM_ROWS = 5

    # 1. MAJOR EVENTS & TRANSFERS (The "Physical" Story)
    transfers = pd.read_csv('../data_samples/hosp/transfers.csv')[:NUM_ROWS]
    admissions = pd.read_csv('../data_samples/hosp/admissions.csv')[:NUM_ROWS]
    services = pd.read_csv('../data_samples/hosp/services.csv')[:NUM_ROWS]
    
    # Join transfers with admissions to get the overall context
    events_df = transfers.merge(admissions[['hadm_id', 'admission_type', 'admission_location', 'discharge_location']], on='hadm_id', how='left')
    # Merge with services to see the specialty (Cardiology, Surgery, etc.)
    events_df = events_df.merge(services[['hadm_id', 'curr_service', 'transfertime']], 
                                left_on=['hadm_id'], right_on=['hadm_id'], how='left')
    
    events_df['timestamp'] = pd.to_datetime(events_df['intime'])
    

    # 2. MEDICATIONS (The "Treatment" Story)
    prescriptions = pd.read_csv('../data_samples/hosp/prescriptions.csv')[:NUM_ROWS]
    pharmacy = pd.read_csv('../data_samples/hosp/pharmacy.csv')[:NUM_ROWS]
    
    # Join to get generic names and routes
    meds_df = prescriptions.merge(pharmacy[['pharmacy_id', 'route']], on='pharmacy_id', how='left')
    
    # Use starttime as the anchor for when a treatment was initiated
    meds_df['timestamp'] = pd.to_datetime(meds_df['starttime'])


    # 3. EXAMINATIONS & LABS (The "Objective" Story)
    lab_events = pd.read_csv('../data_samples/hosp/labevents.csv')[:NUM_ROWS]
    d_labitems = pd.read_csv('../data_samples/hosp/d_labitems.csv')[:NUM_ROWS]
    
    # Enrich with human-readable labels
    labs_df = lab_events.merge(d_labitems[['itemid', 'label', 'fluid', 'category']], on='itemid', how='left')
    
    # Use charttime for labs (the actual time the specimen was taken)
    labs_df['timestamp'] = pd.to_datetime(labs_df['charttime'])
    

    # 4. VITALS & BODY MEASURES (The "OMR" Story)
    # omr = pd.read_csv('../data_samples/hosp/omr.csv')
    # omr['timestamp'] = pd.to_datetime(omr['charttime'])



    # 5. PROCEDURES (The "Action" Paragraph)
    
    # A. Bedside/ICU Procedures (Timed events)
    proc_events = pd.read_csv('../data_samples/icu/procedureevents.csv')
    d_items = pd.read_csv('../data_samples/icu/d_items.csv')
    
    icu_procs = proc_events.merge(d_items[['itemid', 'label', 'category']], on='itemid', how='left')
    icu_procs['timestamp'] = pd.to_datetime(icu_procs['starttime'])
    icu_procs['endtime'] = pd.to_datetime(icu_procs['endtime'])

    # B. Billed/Hospital Procedures (The "Translation" fix)
    hosp_procs = pd.read_csv('../data_samples/hosp/procedures_icd.csv')
    d_icd_procs = pd.read_csv('../data_samples/hosp/d_icd_procedures.csv')
    
    # --- CLEANING STEP: This ensures "5491" matches "5491" regardless of types ---
    for df in [hosp_procs, d_icd_procs]:
        df['icd_code'] = df['icd_code'].astype(str).str.strip()
        df['icd_version'] = df['icd_version'].astype(int)

    # Now join to get names like "Percutaneous abdominal drainage"
    billed_procs = hosp_procs.merge(
        d_icd_procs[['icd_code', 'icd_version', 'long_title']], 
        on=['icd_code', 'icd_version'], 
        how='left'
    )
    
    # Merge with admissions to get the time anchor
    admissions = pd.read_csv('../data_samples/hosp/admissions.csv')
    billed_procs = billed_procs.merge(admissions[['hadm_id', 'admittime']], on='hadm_id', how='left')
    billed_procs['timestamp'] = pd.to_datetime(billed_procs['admittime'])


    events_df.to_csv('events_df.csv', index=False)
    meds_df.to_csv('meds_df.csv', index=False)
    labs_df.to_csv('labs_df.csv', index=False)

    return {
        'events': events_df,
        'meds': meds_df,
        'labs': labs_df,
        # 'omr': omr,
        'icu_procedures': icu_procs,
        'billed_procedures': billed_procs
    }

def load_data():
    discharges = load_discharges()
    patient_data = load_patient_data()
    merged_data = pd.merge(discharges, patient_data, on='subject_id', how='left')
    return merged_data


print(load_clinical_context())