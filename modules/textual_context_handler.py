import pandas as pd


def textual_context_builder():
    return



def data_preparation(subject_ids=None):
    """
    subject_ids: List of integers. If None, it processes everyone (not recommended for RAM).
    """
    # 1. Load Discharges (Filtered)
    # Assuming load_discharges is updated to accept IDs
    discharges = load_discharges(subject_ids).sort_values(['subject_id', 'storetime'])
    
    # 2. Load Clinical Context (Filtered)
    clinical_data_dict = load_clinical_context(subject_ids) 

    cases_data = []

    # Groupby is now much faster because clinical_data_dict only contains relevant rows
    for subject_id, group in discharges.groupby('subject_id'):
        subj_context = {k: df[df['subject_id'] == subject_id] for k, df in clinical_data_dict.items()}

        for case in group.itertuples(index=False):
            window_start = case.window_start
            window_end   = case.storetime

            window_cuts = {}
            for category, df in subj_context.items():
                # Temporal slicing
                window_cuts[category] = df[
                    (df['timestamp'] >= window_start) & 
                    (df['timestamp'] <= window_end)
                ]

            context_textual = textual_context_builder(window_cuts)

            cases_data.append({
                'note_id': case.note_id,
                'subject_id': subject_id,
                'hadm_id': case.hadm_id,
                'discharge_text': case.text,
                'context_textual': context_textual
            })

    return pd.DataFrame(cases_data)


def filtered_read_csv(file_path, subject_ids, **kwargs):
    """Helper to read large CSVs in chunks and filter by subject_id."""
    chunk_list = []
    # Adjust chunksize based on your RAM; 100k is usually safe
    for chunk in pd.read_csv(file_path, chunksize=100_000, **kwargs):
        filtered_chunk = chunk[chunk['subject_id'].isin(subject_ids)]
        chunk_list.append(filtered_chunk)
    return pd.concat(chunk_list)

def load_identity_pillar(subject_ids):
    # A. Load core tables with filtering
    admissions = filtered_read_csv('admissions.csv', subject_ids)
    patients = filtered_read_csv('patients.csv', subject_ids)
    omr = filtered_read_csv('omr.csv', subject_ids)

    # 1. Standardize Timestamps
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    patients['anchor_year'] = patients['anchor_year'].astype(int)

    # 2. Merge Admissions and Patients
    identity_df = admissions.merge(
        patients[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod']], 
        on='subject_id', 
        how='left'
    )

    # 3. Calculate Age at Admission
    identity_df['admission_year'] = identity_df['admittime'].dt.year
    identity_df['age_at_admission'] = (identity_df['anchor_age'] + 
                                       (identity_df['admission_year'] - identity_df['anchor_year']))

    # 4. Integrate OMR (Baseline Vitals)
    relevant_omr_names = ['Weight (Lbs)', 'Height (Inches)', 'BMI (kg/m2)', 'Blood Pressure']
    baseline_omr = omr[omr['result_name'].isin(relevant_omr_names)].copy()
    
    # Sort and get first recorded value per subject/measure
    baseline_omr = baseline_omr.sort_values('chartdate').groupby(['subject_id', 'result_name']).head(1)

    # Pivot
    omr_pivot = baseline_omr.pivot(index='subject_id', columns='result_name', values='result_value').reset_index()

    # Merge OMR data
    identity_df = identity_df.merge(omr_pivot, on='subject_id', how='left')

    # 5. Set the Anchor Timestamp & Cleanup
    identity_df['timestamp'] = identity_df['admittime']
    
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
    
    # Ensure all columns exist before selecting (handles cases where OMR might be missing a column)
    existing_cols = [c for c in relevant_cols if c in identity_df.columns]
    return identity_df[existing_cols]


def load_logistics_pillar(subject_ids):
    """
    Chunks and filters the transfers, services, and admissions tables 
    to build the logistics timeline for specific subjects.
    """
    # 0. Filtered Reads (Assuming HOSP is your file path prefix)
    transfers = filtered_read_csv(HOSP + 'transfers.csv', subject_ids)
    services = filtered_read_csv(HOSP + 'services.csv', subject_ids)
    admissions = filtered_read_csv(HOSP + 'admissions.csv', subject_ids)

    # 1. Standardize Timestamps
    transfers['intime'] = pd.to_datetime(transfers['intime'])
    transfers['outime'] = pd.to_datetime(transfers['outtime'])
    services['transfertime'] = pd.to_datetime(services['transfertime'])

    # 2. Cleanup and Type Casting
    # Removing NaNs in hadm_id is crucial for the asof_merge and standard merges
    transfers = transfers.dropna(subset=['hadm_id'])
    services = services.dropna(subset=['hadm_id'])
    admissions = admissions.dropna(subset=['hadm_id'])

    transfers['hadm_id'] = transfers['hadm_id'].astype('int64')
    services['hadm_id'] = services['hadm_id'].astype('int64')
    admissions['hadm_id'] = admissions['hadm_id'].astype('int64')

    # 3. Enrich Transfers with Admission/Discharge info
    # We only take the columns we need from admissions to save RAM
    logistics_df = transfers.merge(
        admissions[['hadm_id', 'admission_location', 'discharge_location']], 
        on='hadm_id', 
        how='left'
    )

    # 4. Join Services (The "As-Of" Join)
    # asof join requires both dataframes to be sorted by the key
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
    # We filter out rows without careunits (like administrative rows)
    logistics_df = logistics_df.dropna(subset=['careunit'])
    logistics_df['stay_duration_hours'] = (
        (logistics_df['outime'] - logistics_df['intime']).dt.total_seconds() / 3600
    )

    # 6. Set the Anchor Timestamp for temporal windowing
    logistics_df['timestamp'] = logistics_df['intime']

    relevant_cols = [
        'subject_id', 'hadm_id', 'timestamp', 'eventtype', 'careunit', 
        'curr_service', 'intime', 'outime', 'stay_duration_hours',
        'admission_location', 'discharge_location'
    ]

    # Use only columns that successfully merged/exist
    existing_cols = [c for c in relevant_cols if c in logistics_df.columns]
    
    return logistics_df[existing_cols]


def load_monitoring_pillar(subject_ids):
    """
    Chunks and filters chartevents (ICU) and OMR (Ward) to build a 
    comprehensive vitals timeline.

    Pro-Tip for this Pillar:
    Because chartevents is so massive, I increased the chunksize to 200,000. If you find it's still slow, you can use the usecols parameter in pd.read_csv inside filtered_read_csv to only load ['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum', 'valueuom']. This prevents Python from even looking at the other 10+ columns in that table, saving massive amounts of overhead.
    """
    # 1. Filter d_items first (Small table, can stay in RAM)
    # 220045: Heart Rate, 220179/80: BP, 220210: RR, 223761: Temp
    vital_itemids = [220045, 220179, 220180, 220210, 223761]
    d_items = pd.read_csv(ICU + 'd_items.csv')
    vitals_dict = d_items[d_items['itemid'].isin(vital_itemids)][['itemid', 'label']]

    # 2. Join ICU Vitals (The "Big One")
    # We use filtered_read_csv with an inner join inside the loop logic
    # to only keep the relevant itemids for our cohort.
    icu_vitals_list = []
    for chunk in pd.read_csv(ICU + 'chartevents.csv', chunksize=200_000):
        # Filter by subject AND itemid immediately
        f_chunk = chunk[(chunk['subject_id'].isin(subject_ids)) & 
                        (chunk['itemid'].isin(vital_itemids))]
        if not f_chunk.empty:
            icu_vitals_list.append(f_chunk)
    
    if icu_vitals_list:
        icu_vitals = pd.concat(icu_vitals_list)
        icu_vitals = icu_vitals.merge(vitals_dict, on='itemid', how='inner')
        icu_vitals['timestamp'] = pd.to_datetime(icu_vitals['charttime'])
    else:
        icu_vitals = pd.DataFrame(columns=['subject_id', 'hadm_id', 'timestamp', 'label', 'valuenum', 'valueuom'])

    # 3. Join Ward Vitals (from OMR)
    # We need admissions to map OMR back to hadm_id
    omr = filtered_read_csv(HOSP + 'omr.csv', subject_ids)
    admissions = filtered_read_csv(HOSP + 'admissions.csv', subject_ids)
    
    ward_vitals = omr[omr['result_name'].str.contains('Blood Pressure', na=False)].copy()
    ward_vitals['timestamp'] = pd.to_datetime(ward_vitals['chartdate'])

    # Link OMR to admission windows
    ward_vitals = ward_vitals.merge(
        admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime']], 
        on='subject_id', 
        how='left'
    )

    # Convert to datetime and filter for records during admission
    ward_vitals['admittime'] = pd.to_datetime(ward_vitals['admittime'])
    ward_vitals['dischtime'] = pd.to_datetime(ward_vitals['dischtime'])

    mask = (ward_vitals['timestamp'] >= ward_vitals['admittime'].dt.normalize()) & \
           (ward_vitals['timestamp'] <= ward_vitals['dischtime'].dt.normalize())
    ward_vitals = ward_vitals[mask]
    ward_vitals = ward_vitals.rename(columns={'result_name': 'label', 'result_value': 'valuenum'})

    # 4. Combine into a "Master Vitals" table
    monitoring_df = pd.concat([
        icu_vitals[['subject_id', 'hadm_id', 'timestamp', 'label', 'valuenum', 'valueuom']],
        ward_vitals[['subject_id', 'hadm_id', 'timestamp', 'label', 'valuenum']]
    ], axis=0, ignore_index=True)

    return monitoring_df.sort_values(['subject_id', 'timestamp'])


def load_investigations_pillar(subject_ids):
    """
    Chunks and filters labevents and microbiologyevents to build
    the investigations (Labs & Micro) timeline.
    """
    # 1. Load Lab Dictionaries (Small, kept in RAM)
    d_labitems = pd.read_csv(HOSP + 'd_labitems.csv')
    
    # 2. Process Lab Events (The "Big One")
    # Using usecols to only pull what we absolutely need from the CSV
    lab_cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum', 'valueuom', 'flag']
    
    labs_list = []
    for chunk in pd.read_csv(HOSP + 'labevents.csv', chunksize=200_000, usecols=lab_cols):
        # Filter by subject_ids immediately
        f_chunk = chunk[chunk['subject_id'].isin(subject_ids)]
        if not f_chunk.empty:
            labs_list.append(f_chunk)
            
    if labs_list:
        labs_df = pd.concat(labs_list)
        labs_df['timestamp'] = pd.to_datetime(labs_df['charttime'])
        
        # Merge with dictionary for labels/fluids
        labs_final = labs_df.merge(
            d_labitems[['itemid', 'label', 'fluid', 'category']], 
            on='itemid', 
            how='left'
        )
        labs_final['type'] = 'LAB'
    else:
        labs_final = pd.DataFrame(columns=['subject_id', 'hadm_id', 'timestamp', 'label', 'valuenum', 'valueuom', 'flag', 'fluid', 'type'])

    # 3. Process Microbiology (Usually smaller, but chunking for safety)
    micro_list = []
    for chunk in pd.read_csv(HOSP + 'microbiologyevents.csv', chunksize=100_000):
        f_chunk = chunk[chunk['subject_id'].isin(subject_ids)]
        if not f_chunk.empty:
            micro_list.append(f_chunk)
            
    if micro_list:
        micro_df = pd.concat(micro_list)
        micro_df['timestamp'] = pd.to_datetime(micro_df['charttime'])
        
        micro_final = micro_df[[
            'subject_id', 'hadm_id', 'timestamp', 'spec_type_desc', 
            'org_name', 'ab_name', 'interpretation'
        ]].copy()
        
        micro_final = micro_final.rename(columns={'spec_type_desc': 'label'})
        micro_final['type'] = 'MICRO'
    else:
        micro_final = pd.DataFrame(columns=['subject_id', 'hadm_id', 'timestamp', 'label', 'org_name', 'ab_name', 'interpretation', 'type'])

    # 4. Combine into Investigations Table
    investigations_df = pd.concat([labs_final, micro_final], axis=0, ignore_index=True)
    
    return investigations_df.sort_values(['subject_id', 'timestamp'])


def load_interventions_pillar(subject_ids):
    """
    Chunks and filters procedureevents (ICU) and procedures_icd (HOSP)
    to build the interventions and surgical history timeline.
    """
    # 1. Load Dictionaries (Small tables)
    d_items = pd.read_csv(ICU + 'd_items.csv')
    d_icd_procedures = pd.read_csv(HOSP + 'd_icd_procedures.csv')
    
    # Standardize Dictionary for ICD Join
    d_icd_procedures['icd_version'] = d_icd_procedures['icd_version'].astype(int)
    d_icd_procedures['icd_code'] = d_icd_procedures['icd_code'].astype(str).str.strip()
    mask_v9_dict = d_icd_procedures['icd_version'] == 9
    d_icd_procedures.loc[mask_v9_dict, 'icd_code'] = d_icd_procedures.loc[mask_v9_dict, 'icd_code'].str.zfill(4)

    # 2. ICU Bedside Procedures
    icu_procs_list = []
    for chunk in pd.read_csv(ICU + 'procedureevents.csv', chunksize=100_000):
        f_chunk = chunk[chunk['subject_id'].isin(subject_ids)]
        if not f_chunk.empty:
            icu_procs_list.append(f_chunk)
            
    if icu_procs_list:
        icu_procs = pd.concat(icu_procs_list)
        icu_procs['timestamp'] = pd.to_datetime(icu_procs['starttime'])
        icu_procs['endtime'] = pd.to_datetime(icu_procs['endtime'])
        icu_procs = icu_procs.merge(d_items[['itemid', 'label']], on='itemid', how='left')
        icu_procs['proc_type'] = 'ICU_BEDSIDE'
    else:
        icu_procs = pd.DataFrame(columns=['subject_id', 'hadm_id', 'timestamp', 'label', 'proc_type', 'endtime'])

    # 3. Billed Hospital Procedures
    billed_procs_list = []
    for chunk in pd.read_csv(HOSP + 'procedures_icd.csv', chunksize=100_000):
        f_chunk = chunk[chunk['subject_id'].isin(subject_ids)]
        if not f_chunk.empty:
            billed_procs_list.append(f_chunk)
            
    if billed_procs_list:
        billed_procs = pd.concat(billed_procs_list)
        
        # Standardize ICD codes
        billed_procs['icd_version'] = billed_procs['icd_version'].astype(int)
        billed_procs['icd_code'] = billed_procs['icd_code'].astype(str).str.strip()
        mask_v9_billed = billed_procs['icd_version'] == 9
        billed_procs.loc[mask_v9_billed, 'icd_code'] = billed_procs.loc[mask_v9_billed, 'icd_code'].str.zfill(4)
        
        # Merge with dictionary
        billed_procs = billed_procs.merge(
            d_icd_procedures[['icd_code', 'icd_version', 'long_title']], 
            on=['icd_code', 'icd_version'], 
            how='left'
        )
        
        # Merge with admissions to get a timestamp (since billed procs only have chartdate)
        # Note: We can reuse the filtered_read_csv or a smaller subset of admissions
        admissions = filtered_read_csv(HOSP + 'admissions.csv', subject_ids)
        billed_procs = billed_procs.merge(admissions[['hadm_id', 'admittime']], on='hadm_id', how='left')
        
        billed_procs['timestamp'] = pd.to_datetime(billed_procs['admittime'])
        billed_procs = billed_procs.rename(columns={'long_title': 'label'})
        billed_procs['proc_type'] = 'BILLED_SURGICAL'
    else:
        billed_procs = pd.DataFrame(columns=['subject_id', 'hadm_id', 'timestamp', 'label', 'proc_type'])

    # 4. Combine into Interventions Table
    interventions_df = pd.concat([
        icu_procs[['subject_id', 'hadm_id', 'timestamp', 'label', 'proc_type', 'endtime']],
        billed_procs[['subject_id', 'hadm_id', 'timestamp', 'label', 'proc_type']]
    ], axis=0, ignore_index=True)

    return interventions_df.sort_values(['subject_id', 'timestamp'])


def load_inputs_pillar(subject_ids):
    """
    Chunks and filters medication data from ICU inputevents and 
    Hospital prescriptions for specific subjects.
    """
    # 1. Load Dictionary (Small table)
    d_items = pd.read_csv(ICU + 'd_items.csv')

    # 2. ICU Continuous Inputs (IVs and Drips)
    # Define only necessary columns to minimize memory footprint during read
    icu_cols = ['subject_id', 'hadm_id', 'itemid', 'starttime', 'amount', 'amountuom', 'rate', 'rateuom']
    
    icu_inputs_list = []
    for chunk in pd.read_csv(ICU + 'inputevents.csv', chunksize=150_000, usecols=icu_cols):
        f_chunk = chunk[chunk['subject_id'].isin(subject_ids)]
        if not f_chunk.empty:
            icu_inputs_list.append(f_chunk)
            
    if icu_inputs_list:
        icu_inputs = pd.concat(icu_inputs_list)
        icu_inputs['timestamp'] = pd.to_datetime(icu_inputs['starttime'])
        
        # Join with d_items to get drug names
        icu_inputs = icu_inputs.merge(d_items[['itemid', 'label']], on='itemid', how='left')
        icu_inputs['input_type'] = 'ICU_INPUT'
        
        # Select final columns for this sub-section
        icu_inputs = icu_inputs[['subject_id', 'hadm_id', 'timestamp', 'label', 'amount', 
                                 'amountuom', 'rate', 'rateuom', 'input_type']]
    else:
        icu_inputs = pd.DataFrame(columns=['subject_id', 'hadm_id', 'timestamp', 'label', 'input_type'])

    # 3. General Hospital Prescriptions
    rx_cols = ['subject_id', 'hadm_id', 'starttime', 'drug', 'dose_val_rx', 'dose_unit_rx', 'route']
    
    ward_inputs_list = []
    for chunk in pd.read_csv(HOSP + 'prescriptions.csv', chunksize=200_000, usecols=rx_cols):
        f_chunk = chunk[chunk['subject_id'].isin(subject_ids)]
        if not f_chunk.empty:
            ward_inputs_list.append(f_chunk)
            
    if ward_inputs_list:
        ward_inputs = pd.concat(ward_inputs_list)
        ward_inputs['timestamp'] = pd.to_datetime(ward_inputs['starttime'])
        
        # Rename and cast types
        ward_inputs = ward_inputs.rename(columns={
            'drug': 'label',
            'dose_val_rx': 'amount',
            'dose_unit_rx': 'amountuom'
        })
        ward_inputs['input_type'] = 'WARD_PRESCRIPTION'
    else:
        ward_inputs = pd.DataFrame(columns=['subject_id', 'hadm_id', 'timestamp', 'label', 'input_type'])

    # 4. Combine into a Master Meds Table
    meds_df = pd.concat([icu_inputs, ward_inputs], axis=0, ignore_index=True)
    
    return meds_df.sort_values(['subject_id', 'timestamp'])


def load_conclusion_pillar(subject_ids):
    """
    Chunks and filters diagnoses_icd and joins with d_icd_diagnoses 
    to provide the final clinical summary of the admission.
    """
    # 1. Load Dictionary (Small table)
    d_icd_diagnoses = pd.read_csv(HOSP + 'd_icd_diagnoses.csv')
    
    # Clean and Pad Dictionary ICD Codes
    d_icd_diagnoses['icd_version'] = d_icd_diagnoses['icd_version'].astype(int)
    d_icd_diagnoses['icd_code'] = d_icd_diagnoses['icd_code'].astype(str).str.strip()
    mask_v9_dict = d_icd_diagnoses['icd_version'] == 9
    d_icd_diagnoses.loc[mask_v9_dict, 'icd_code'] = d_icd_diagnoses.loc[mask_v9_dict, 'icd_code'].str.zfill(3)

    # 2. Process Diagnoses (Chunked)
    diagnoses_list = []
    for chunk in pd.read_csv(HOSP + 'diagnoses_icd.csv', chunksize=100_000):
        f_chunk = chunk[chunk['subject_id'].isin(subject_ids)]
        if not f_chunk.empty:
            diagnoses_list.append(f_chunk)
            
    if diagnoses_list:
        diagnoses_df = pd.concat(diagnoses_list)
        
        # Clean and Pad Event ICD Codes
        diagnoses_df['icd_version'] = diagnoses_df['icd_version'].astype(int)
        diagnoses_df['icd_code'] = diagnoses_df['icd_code'].astype(str).str.strip()
        mask_v9_event = diagnoses_df['icd_version'] == 9
        diagnoses_df.loc[mask_v9_event, 'icd_code'] = diagnoses_df.loc[mask_v9_event, 'icd_code'].str.zfill(3)

        # Join with Dictionary
        outcomes_df = diagnoses_df.merge(
            d_icd_diagnoses[['icd_code', 'icd_version', 'long_title']], 
            on=['icd_code', 'icd_version'], 
            how='left'
        )

        # 3. Anchor to Discharge Time
        # Re-using the filtered admissions reader to get dischtime
        admissions = filtered_read_csv(HOSP + 'admissions.csv', subject_ids)
        outcomes_df = outcomes_df.merge(
            admissions[['hadm_id', 'dischtime']], 
            on='hadm_id', 
            how='left'
        )

        outcomes_df['timestamp'] = pd.to_datetime(outcomes_df['dischtime'])
        outcomes_df = outcomes_df.rename(columns={'long_title': 'diagnosis_label'})
        
        relevant_cols = ['subject_id', 'hadm_id', 'timestamp', 'diagnosis_label', 'seq_num']
        return outcomes_df[relevant_cols].sort_values(['subject_id', 'seq_num'])
    
    else:
        return pd.DataFrame(columns=['subject_id', 'hadm_id', 'timestamp', 'diagnosis_label', 'seq_num'])



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


def load_discharges():
    data = pd.read_csv('../data_samples/notes/discharge.csv')
    return data[:5] ### TODO: Remove limit in the return