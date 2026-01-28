import os

import pandas as pd

HOSP = '../data_samples/hosp/'
ICU = '../data_samples/icu/'

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


def load_clinical_context_xxx():

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


def load_identity_pillar():

    patients = pd.read_csv(HOSP+'patients.csv')
    admissions = pd.read_csv(HOSP+'admissions.csv')
    omr = pd.read_csv(HOSP+'omr.csv')

    # 1. Standardize Timestamps
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    patients['anchor_year'] = patients['anchor_year'].astype(int)

    # 2. Merge Admissions and Patients
    # We bring in gender and the age anchors
    identity_df = admissions.merge(
        patients[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod']], 
        on='subject_id', 
        how='left'
    )

    # 3. Calculate Age at Admission
    # Age = anchor_age + (Year of Admission - anchor_year)
    identity_df['admission_year'] = identity_df['admittime'].dt.year
    identity_df['age_at_admission'] = (identity_df['anchor_age'] + (identity_df['admission_year'] - identity_df['anchor_year']))

    # 4. Integrate OMR (Baseline Vitals)
    # Height, Weight, BMI, Blood pressure
    baseline_omr = omr[omr['result_name'].isin(['Weight (Lbs)', 'Height (Inches)', 'BMI (kg/m2)', 'Blood Pressure'])].copy()
    baseline_omr = baseline_omr.sort_values('chartdate').groupby(['subject_id', 'result_name']).head(1)

    # Pivot so each measurement is its own column
    omr_pivot = baseline_omr.pivot(index='subject_id', columns='result_name', values='result_value').reset_index()

    # Merge OMR data into main identity frame
    identity_df = identity_df.merge(omr_pivot, on='subject_id', how='left')

    # 5. Set the Anchor Timestamp
    identity_df['timestamp'] = identity_df['admittime']

    # 6. Cleanup and Selection
    # Rename OMR columns to be more regex-friendly (remove spaces/units if preferred)
    identity_df = identity_df.rename(columns={
        'Weight (Lbs)': 'weight_baseline',
        'Height (Inches)': 'height_baseline',
        'BMI (kg/m2)': 'bmi_baseline',
        'Blood Pressure': 'blood_pressure_baseline',
    })

    relevant_cols = [
        'subject_id', 'hadm_id', 'timestamp', 'age_at_admission', 'gender', 'race',
        'admission_type', 'admission_location', 'insurance', 'marital_status',
        'weight_baseline', 'height_baseline', 'bmi_baseline', 'blood_pressure_baseline', 'dod'
    ]

    identity_df = identity_df[relevant_cols]
    identity_df = identity_df.sort_values('timestamp')
    return identity_df



def load_logistics_pillar():

    transfers = pd.read_csv(HOSP+'transfers.csv')
    services = pd.read_csv(HOSP+'services.csv')
    admissions = pd.read_csv(HOSP+'admissions.csv')

    # 1. Standardize Timestamps
    transfers['intime'] = pd.to_datetime(transfers['intime'])
    transfers['outime'] = pd.to_datetime(transfers['outtime'])
    services['transfertime'] = pd.to_datetime(services['transfertime'])

    # 2. Fix HADM_ID types to avoid MergeError
    # We drop NaNs first because you can't join on an empty ID
    transfers = transfers.dropna(subset=['hadm_id'])
    services = services.dropna(subset=['hadm_id'])

    # Force to int64
    transfers['hadm_id'] = transfers['hadm_id'].astype('int64')
    services['hadm_id'] = services['hadm_id'].astype('int64')
    admissions['hadm_id'] = admissions['hadm_id'].astype('int64')

    # 3. Enrich Transfers with Admission/Discharge info
    logistics_df = transfers.merge(
        admissions[['hadm_id', 'admission_location', 'discharge_location']], 
        on='hadm_id', 
        how='left'
    )

    # 4. Join Services (The "As-Of" Join)
    services = services.sort_values('transfertime')
    logistics_df = logistics_df.sort_values('intime')

    logistics_df = pd.merge_asof(
        logistics_df,
        services[['hadm_id', 'curr_service', 'transfertime']],
        left_on='intime',
        right_on='transfertime',
        by='hadm_id',
        direction='backward'
    )

    # 5. Filter for real ward movements and calculate duration
    logistics_df = logistics_df.dropna(subset=['careunit'])
    logistics_df['stay_duration_hours'] = (logistics_df['outime'] - logistics_df['intime']).dt.total_seconds() / 3600

    # 6. Set the Anchor Timestamp
    logistics_df['timestamp'] = logistics_df['intime']

    relevant_cols = [
        'subject_id', 'hadm_id', 'timestamp', 'eventtype', 'careunit', 
        'curr_service', 'intime', 'outime', 'stay_duration_hours',
        'admission_location', 'discharge_location'
    ]

    logistics_df = logistics_df[relevant_cols]
    logistics_df = logistics_df.sort_values('timestamp')
    return logistics_df


def load_monitoring_pillar():

    d_items = pd.read_csv(ICU+'d_items.csv')
    chartevents = pd.read_csv(ICU+'chartevents.csv')
    admissions = pd.read_csv(HOSP+'admissions.csv')
    omr = pd.read_csv(HOSP+'omr.csv')

    # 1. Filter d_items for common Vital Signs to keep the data manageable
    # In MIMIC-IV, these are the most common ICU vital codes
    vital_itemids = [220045, 220179, 220180, 220210, 223761] # HR, SysBP, DiasBP, RR, Temp
    vitals_dict = d_items[d_items['itemid'].isin(vital_itemids)]

    # 2. Join ICU Vitals
    icu_vitals = chartevents.merge(vitals_dict[['itemid', 'label']], on='itemid', how='inner')
    icu_vitals['timestamp'] = pd.to_datetime(icu_vitals['charttime'])

    # 3. Join Ward Vitals (from OMR)
    ward_vitals = omr[omr['result_name'].str.contains('Blood Pressure', na=False)].copy()
    ward_vitals['timestamp'] = pd.to_datetime(ward_vitals['chartdate'])

    # Merge with admissions to get the hadm_id for each OMR measurement
    # Note: Since OMR doesn't have hadm_id, we link via subject_id and time
    ward_vitals = ward_vitals.merge(admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime']], on='subject_id', how='left')

    # Filter to keep only OMR records that happened DURING the admission
    ward_vitals['admittime'] = pd.to_datetime(ward_vitals['admittime'])
    ward_vitals['dischtime'] = pd.to_datetime(ward_vitals['dischtime'])

    mask = (ward_vitals['timestamp'] >= ward_vitals['admittime'].dt.normalize()) & \
        (ward_vitals['timestamp'] <= ward_vitals['dischtime'].dt.normalize())
    ward_vitals = ward_vitals[mask]

    # Now rename and select columns
    ward_vitals = ward_vitals.rename(columns={'result_name': 'label', 'result_value': 'valuenum'})

    # 4. Combine into a "Master Vitals" table
    # Now ward_vitals has 'hadm_id', so the concatenation won't crash
    monitoring_df = pd.concat([
        icu_vitals[['subject_id', 'hadm_id', 'timestamp', 'label', 'valuenum', 'valueuom']],
        ward_vitals[['subject_id', 'hadm_id', 'timestamp', 'label', 'valuenum']]
    ], axis=0)

    monitoring_df = monitoring_df.sort_values('timestamp')
    return monitoring_df


def load_investigations_pillar():
    labevents = pd.read_csv(HOSP+'labevents.csv')
    d_labitems = pd.read_csv(HOSP+'d_labitems.csv')
    microbiology = pd.read_csv(HOSP+'microbiologyevents.csv')

    # 1. Standardize Lab Events
    labevents['timestamp'] = pd.to_datetime(labevents['charttime'])

    # Merge with dictionary to get names (Labels) and the specimen type (Fluid)
    labs_df = labevents.merge(
        d_labitems[['itemid', 'label', 'fluid', 'category']], 
        on='itemid', 
        how='left'
    )

    # Select key columns for the summary
    # 'flag' is vital here: it tells us if a result was 'abnormal'
    labs_final = labs_df[[
        'subject_id', 'hadm_id', 'timestamp', 'label', 'valuenum', 
        'valueuom', 'flag', 'fluid'
    ]].copy()
    labs_final['type'] = 'LAB'

    # 2. Process Microbiology
    microbiology['timestamp'] = pd.to_datetime(microbiology['charttime'])

    # We focus on the specimen (what was tested) and the organism (what was found)
    # If org_name is NaN, it usually means "No growth"
    micro_final = microbiology[[
        'subject_id', 'hadm_id', 'timestamp', 'spec_type_desc', 
        'org_name', 'ab_name', 'interpretation'
    ]].copy()

    micro_final = micro_final.rename(columns={'spec_type_desc': 'label'})
    micro_final['type'] = 'MICRO'

    # 3. Combine into the Investigations Table
    investigations_df = pd.concat([labs_final, micro_final], axis=0)

    investigations_df = investigations_df.sort_values('timestamp')
    return investigations_df


def load_interventions_pillar():
    # Procedures
    procedureevents = pd.read_csv(ICU+'procedureevents.csv')
    d_items = pd.read_csv(ICU+'d_items.csv')
    admissions = pd.read_csv(HOSP+'admissions.csv')

    d_icd_procedures = pd.read_csv(HOSP+'d_icd_procedures.csv')
    procedures_icd = pd.read_csv(HOSP+'procedures_icd.csv')


    # 1. ICU Bedside Procedures
    procedureevents['timestamp'] = pd.to_datetime(procedureevents['starttime'])
    procedureevents['endtime'] = pd.to_datetime(procedureevents['endtime'])

    icu_procs = procedureevents.merge(d_items[['itemid', 'label']], on='itemid', how='left')
    icu_procs['proc_type'] = 'ICU_BEDSIDE'

    # 2. Billed Hospital Procedures (The "Translation" Fix)
    # Force versions to int and codes to padded strings
    for df in [procedures_icd, d_icd_procedures]:
        df['icd_version'] = df['icd_version'].astype(int)
        df['icd_code'] = df['icd_code'].astype(str).str.strip()
        # Pad ICD-9 codes to 4 digits (e.g., '66' -> '0066')
        mask_v9 = df['icd_version'] == 9
        df.loc[mask_v9, 'icd_code'] = df.loc[mask_v9, 'icd_code'].str.zfill(4)

    billed_procs = procedures_icd.merge(
        d_icd_procedures[['icd_code', 'icd_version', 'long_title']], 
        on=['icd_code', 'icd_version'], 
        how='left'
    )

    # Billed procedures only have a 'chartdate'. We anchor them to 'admittime' 
    # so they appear in the stay window, but label them as billed events.
    billed_procs = billed_procs.merge(admissions[['hadm_id', 'admittime']], on='hadm_id', how='left')
    billed_procs['timestamp'] = pd.to_datetime(billed_procs['admittime'])
    billed_procs = billed_procs.rename(columns={'long_title': 'label'})
    billed_procs['proc_type'] = 'BILLED_SURGICAL'

    # 3. Combine into Interventions Table
    interventions_df = pd.concat([
        icu_procs[['subject_id', 'hadm_id', 'timestamp', 'label', 'proc_type', 'endtime']],
        billed_procs[['subject_id', 'hadm_id', 'timestamp', 'label', 'proc_type']]
    ], axis=0)

    interventions_df = interventions_df.sort_values('timestamp')
    return interventions_df


def load_inputs_pillar():
    inputevents = pd.read_csv(ICU+'inputevents.csv')
    d_items = pd.read_csv(ICU+'d_items.csv')

    prescriptions = pd.read_csv(HOSP+'prescriptions.csv')

    # 1. ICU Continuous Inputs (IVs and Drips)
    inputevents['timestamp'] = pd.to_datetime(inputevents['starttime'])

    # Join with d_items to get drug names
    icu_inputs = inputevents.merge(d_items[['itemid', 'label']], on='itemid', how='left')

    # We keep amount and rate to describe the intensity of the treatment
    icu_inputs = icu_inputs[['subject_id', 'hadm_id', 'timestamp', 'label', 'amount', 'amountuom', 'rate', 'rateuom']]
    icu_inputs['input_type'] = 'ICU_INPUT'

    # 2. General Hospital Prescriptions
    prescriptions['timestamp'] = pd.to_datetime(prescriptions['starttime'])

    # We select the drug name, dose, and route (e.g., PO for mouth, IV for vein)
    ward_inputs = prescriptions[[
        'subject_id', 'hadm_id', 'timestamp', 'drug', 'dose_val_rx', 'dose_unit_rx', 'route'
    ]].copy()

    ward_inputs = ward_inputs.rename(columns={'drug': 'label'})
    ward_inputs['input_type'] = 'WARD_PRESCRIPTION'

    # 3. Combine into a Master Meds Table
    meds_df = pd.concat([icu_inputs, ward_inputs], axis=0)
        
    meds_df = meds_df.sort_values('timestamp')
    return meds_df


def load_conclusion_pillar():

    diagnoses_icd = pd.read_csv(HOSP+'diagnoses_icd.csv')
    d_icd_diagnoses = pd.read_csv(HOSP+'d_icd_diagnoses.csv')
    admissions = pd.read_csv(HOSP+'admissions.csv')

    # 1. Clean and Pad ICD Codes (Same logic as Procedures)
    for df in [diagnoses_icd, d_icd_diagnoses]:
        df['icd_version'] = df['icd_version'].astype(int)
        df['icd_code'] = df['icd_code'].astype(str).str.strip()
        # Pad ICD-9 codes to 3-5 digits depending on the code type if necessary
        # Usually, a simple zfill handles the majority of join misses
        mask_v9 = df['icd_version'] == 9
        df.loc[mask_v9, 'icd_code'] = df.loc[mask_v9, 'icd_code'].str.zfill(3)

    # 2. Join with Dictionary
    outcomes_df = diagnoses_icd.merge(
        d_icd_diagnoses[['icd_code', 'icd_version', 'long_title']], 
        on=['icd_code', 'icd_version'], 
        how='left'
    )

    # 3. Anchor to Discharge Time
    outcomes_df = outcomes_df.merge(
        admissions[['hadm_id', 'dischtime']], 
        on='hadm_id', 
        how='left'
    )

    outcomes_df['timestamp'] = pd.to_datetime(outcomes_df['dischtime'])
    outcomes_df = outcomes_df.rename(columns={'long_title': 'diagnosis_label'})

    # 'seq_num' tells us the priority (1 is the primary diagnosis)
    outcomes_df = outcomes_df[['subject_id', 'hadm_id', 'timestamp', 'diagnosis_label', 'seq_num']].sort_values('seq_num')
    return outcomes_df


def load_clinical_context():
    """
    Returns a dictionary of DataFrames, each standardized with a 'timestamp' column
    for temporal windowing.
    """
    context_tables = dict()
    
    # Pillar 1 & 2: Identity & Logistics
    context_tables['identity'] = load_identity_pillar()
    context_tables['logistics'] = load_logistics_pillar()
    
    # Pillar 4: Monitoring (Vitals)
    context_tables['monitoring'] = load_monitoring_pillar()
    
    # Pillar 5: Investigations (Labs & Micro)
    context_tables['investigations'] = load_investigations_pillar()
    
    # Pillar 6: Interventions (Procedures)
    context_tables['interventions'] = load_interventions_pillar()
    
    # Pillar 7: Inputs (Medications)
    context_tables['inputs'] = load_inputs_pillar()
    
    # Pillar 8: Conclusion (Final Diagnoses)
    context_tables['conclusion'] = load_conclusion_pillar()

    return context_tables


print(load_clinical_context())

