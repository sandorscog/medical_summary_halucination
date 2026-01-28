import pandas as pd
from transformers import pipeline
import evaluate
from datetime import datetime
import os
import gc
import traceback

from modules.summarizer import Summarizer
from modules.data_handler import load_discharges, load_patient_data, load_clinical_context
from modules.context_handler import context_builder_text, context_builder_json, build_discharge_windows, get_patient_window_data
from modules.summarizer_loader import load_bart, load_clinical_t5, load_gemini
from modules.textual_context_handler import textual_context_builder


def data_preparation():
    # 1. Load Discharges
    discharges = load_discharges().sort_values(['subject_id', 'storetime'])
    
    # 2. Load Clinical Context (Returns a dict of prepared DataFrames)
    # This dict contains: {'meds': df, 'labs': df, 'events': df}
    clinical_data_dict = load_clinical_context() 

    cases_data = []

    for subject_id, group in discharges.groupby('subject_id'):
        # Pre-filter clinical_data_dict for this specific subject to speed up loops
        subj_context = {k: df[df['subject_id'] == subject_id] for k, df in clinical_data_dict.items()}

        for case in group.itertuples(index=False):
            # Define the Window
            window_start = case.window_start # e.g., previous discharge or Timestamp.min
            window_end   = case.storetime    # The time this note was written

            # Create the "Cuts"
            window_cuts = {}
            for category, df in subj_context.items():
                # Slice the data based on the standardized timestamp
                # Note: 'timestamp' is a column we create during the Load step
                window_cuts[category] = df[
                    (df['timestamp'] >= window_start) & 
                    (df['timestamp'] <= window_end)
                ]

            # Build the textual paragraph using your regex logic
            # This function now takes the dict of cuts
            context_textual = context_builder_v2(window_cuts)

            cases_data.append({
                'note_id': case.note_id,
                'subject_id': subject_id,
                'hadm_id': case.hadm_id,
                'discharge_text': case.text,
                'context_textual': context_textual,
                'raw_cuts': window_cuts # Helpful for debugging
            })

    return pd.DataFrame(cases_data)


def data_preparationxxx():
    """
    Loads discharges, patient data, builds contextual windows inline,
    and prepares tuples of (subject_id, hadm_id, discharge_text, context_text).
    """

    # -------------------------------
    # Load data
    # -------------------------------
    discharges = load_discharges().sort_values(['subject_id', 'storetime']).copy()
    patients_data = load_patient_data().copy()

    # -------------------------------
    # Force datetime columns
    # -------------------------------
    discharges['storetime'] = pd.to_datetime(discharges['storetime'], errors='coerce')

    patients_data['starttime'] = pd.to_datetime(patients_data['starttime'], errors='coerce')
    patients_data['endtime']   = pd.to_datetime(patients_data.get('endtime'), errors='coerce')

    # -------------------------------
    # Build windows
    # -------------------------------
    discharges['window_start'] = (
        discharges.groupby('subject_id')['storetime'].shift(1)
    )

    # If first event: no previous window → very early timestamp
    discharges['window_start'] = discharges['window_start'].fillna(pd.Timestamp.min)

    discharges['window_end'] = discharges['storetime']

    cases_data = []

    # -------------------------------
    # Iterate subjects only once
    # -------------------------------
    for subject_id, group in discharges.groupby('subject_id'):
        for case in group.itertuples(index=False):

            hadm_id = case.hadm_id
            original_text = case.text

            # Window values already guaranteed as Timestamp
            window_start = pd.to_datetime(case.window_start, errors="coerce")
            window_end   = pd.to_datetime(case.window_end, errors="coerce")

            # -----------------------------------------
            # Filter ALL patient data inside time window
            # -----------------------------------------
            context_df = patients_data[  ## TODO: REMOVE
                (patients_data['subject_id'] == subject_id)
                & (patients_data['starttime'] >= window_start)
                & (patients_data['starttime'] <= window_end)
            ]

            # -----------------------------------------
            # Build JSON + text versions of the context
            # -----------------------------------------
            context_json = context_builder_json(context_df) ## TODO: Can be discontinued
            context_textual = context_builder_text(context_json) ## TODO: Change the logic to receive a dictionary of tables

            cases_data.append(
                (case.note_id, subject_id, hadm_id, original_text, context_json, context_textual)
            )

    # -------------------------------
    # Convert to DataFrame and return
    # -------------------------------
    cases_data = pd.DataFrame(
        cases_data,
        columns=['note_id', "subject_id", "hadm_id", "discharge_text", "context_json", "context_textual"]
    )

    return cases_data



# def judge(claims: list[str], discharge_text: str, context_json: dict) -> list[dict]:

#     judged_claims = []
#     for claim in claims:

#         judged_claims.append({
#             'claim': claim,
#             'label': 'entailment REMOVE ME',
#             'score': 0, ## REMOVE THIS
#         })

#     return judged_claims


def claim_corretor(claims: list[dict], discharge_text: str, context_json: dict) -> list[str]:

    kept_claims = []
    for claim in claims:
        if 'CONTRADICTION' in claim['label']:
            continue
        kept_claims.append(claim)

    return kept_claims


def results_eval(original: str, summary: str, enhanced_summary: str) -> dict:
    """Compute ROUGE metrics for base and enhanced summaries."""
    rouge = evaluate.load('rouge')
    rouge_scores = rouge.compute(
        predictions=[summary, enhanced_summary],
        references=[original, original],
        use_aggregator=False
    )

    bertscore = evaluate.load("bertscore")
    bert_scores = bertscore.compute(
        predictions=[summary, enhanced_summary],
        references=[original, original],
        lang="en",
        model_type="bert-base-uncased",
        device="cpu"
    )

    return {
        # ---- ROUGE BASE ----
        "base_rouge1": rouge_scores["rouge1"][0],
        "enhanced_rouge1": rouge_scores["rouge1"][1],
        
        "base_rouge2": rouge_scores["rouge2"][0],
        "enhanced_rouge2": rouge_scores["rouge2"][1],
        
        "base_rougeL": rouge_scores["rougeL"][0],
        "enhanced_rougeL": rouge_scores["rougeL"][1],
        
        "base_rougeLsum": rouge_scores["rougeLsum"][0],
        "enhanced_rougeLsum": rouge_scores["rougeLsum"][1],

        # ---- BERTScore ----
        "base_bertscore_precision": bert_scores["precision"][0],
        "enhanced_bertscore_precision": bert_scores["precision"][1],

        "base_bertscore_recall": bert_scores["recall"][0],
        "enhanced_bertscore_recall": bert_scores["recall"][1],

        "base_bertscore_f1": bert_scores["f1"][0],
        "enhanced_bertscore_f1": bert_scores["f1"][1],
    }


def append_result_to_csv(result_dict: dict, path: str):
    """Append a single result dict to a CSV file, creating it if it doesnt exist."""
    df = pd.DataFrame([result_dict])

    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)


def generate_summaries(discharge: str) -> dict:

    summarizer_01 = Summarizer(model='gemini', temp=.1)
    summarizer_1 = Summarizer(model='gemini', temp=1)
    summarizer_2 = Summarizer(model='gemini', temp=2)

    return {
        'discharge_summ_01': summarizer_01.summ(discharge),
        'discharge_summ_1': summarizer_1.summ(discharge),
        'discharge_summ_2': summarizer_2.summ(discharge),
    }


def generate_summary_database():

    data = data_preparation()

    # Prepare storage
    summ_01_col = []
    summ_1_col = []
    summ_2_col = []

    for _, row in data.iterrows():

        # Get summaries for this discharge
        summaries = generate_summaries(row["discharge_text"])

        summ_01_col.append(summaries["discharge_summ_01"])
        summ_1_col.append(summaries["discharge_summ_1"])
        summ_2_col.append(summaries["discharge_summ_2"])

    # Add as new columns
    data["discharge_summ_01"] = summ_01_col
    data["discharge_summ_1"] = summ_1_col
    data["discharge_summ_2"] = summ_2_col

    # Save
    data.to_csv(f"summaries_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)


def complete_pipeline(save_path: str = "results.csv"):
    """Run full summarization - decomposition - judging - evaluation pipeline."""
    data = data_preparation()
    summarizer = Summarizer(model='gemini')

    for i, case in enumerate(data.itertuples(index=False), start=1): #.iloc[13:]
        subject_id = getattr(case, "subject_id", None)
        hadm_id = getattr(case, "hadm_id", None)
        discharge_text = case.discharge_text
        context_json = case.context_json
        context_text = case.context_textual

        print(f"\nProcessing case {i} - subject {subject_id}, hadm {hadm_id}\n Starting summary...")

        try:
            summary = summarizer.summ(discharge_text)
            print('Decomposing into claims...')
            claims = summarizer.decompose(summary, as_list=True)

            print(f'Processing {len(claims)} claims...')
            # judged_claims = judge(claims, discharge_text, context_json)
            # judged_claims = summarizer.judge(claims, discharge_text, context_text) ## TEST WITH CONTEXTUAL TEXT
            judged_claims = summarizer.LLM_judge(claims, discharge_text, context_text)
            final_claims = claim_corretor(judged_claims, discharge_text, context_json)

            print('Compiling new summary')
            enhanced_summary = summarizer.compile([claim['claim'] for claim in final_claims])

            metrics = results_eval(discharge_text, summary, enhanced_summary)
            llm_metrics = summarizer.llm_judge_metrics(discharge_text, summary, enhanced_summary)


            # Extract motivations for CONTRADICTION-labelled claims
            contradiction_motivations = [
                claim.get("motivation", "")
                for claim in judged_claims
                if "CONTRADICTION" in claim.get("label", "")
            ]

            contradiction_motivation_text = "\n\n---------\n".join(contradiction_motivations)

            result = {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "original_text": discharge_text,
                "context_json": context_json,
                "base_summary": summary,
                "decomposed_claims": "\n".join(claims),
                "judged_claims": "\n".join(f"{jc['claim']} - {jc['label']} - {jc['score']:.3f}" for jc in judged_claims),
                "contradiction_motivations": contradiction_motivation_text,
                "final_claims": "\n".join(claim['claim'] for claim in final_claims),
                "enhanced_summary": enhanced_summary,
                **llm_metrics,
                **metrics,
            }

            append_result_to_csv(result, save_path)
            print(f"Case {i} saved to {save_path}")

        except Exception as e:
            print(f"ERROR!! on case {i}")
            traceback.print_exc()
            continue

    # --- rename final file with timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = f"results_{timestamp}.csv"
    os.rename(save_path, final_path)
    print(f"\n!! Pipeline complete !! Results saved as {final_path}")




def claim_judge_pipeline(save_path=None):
    if save_path is None:
        save_path = f"results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"


    data = [
        {
            "discharge_id": "D12345",
            "discharge": "Paracentesis performed, patient stable for discharge.",
            "context": "56-year-old female with HCV cirrhosis.",
            "temp_0.1": "Patient underwent paracentesis and was discharged stable.",
            "temp_1": "The patient was treated for ascites with paracentesis and improved.",
            "temp_2": "After removing fluid from her abdomen, the patient felt better and went home."
        },
        {
            "discharge_id": "D67890",
            "discharge": "Improved after bronchodilators, discharged with inhalers.",
            "context": "Smoker with chronic COPD.",
            "temp_0.1": "Patient treated for COPD exacerbation and discharged.",
            "temp_1": "The patient improved after bronchodilators and was able to go home.",
            "temp_2": "Breathing treatment helped the patient, who was later discharged."
        },
        {
            "discharge_id": "D54321",
            "discharge": "Improved with lactulose therapy.",
            "context": "HCV cirrhosis with prior HE.",
            "temp_0.1": "Patient treated for hepatic encephalopathy and stabilized.",
            "temp_1": "The patient improved on lactulose for HE.",
            "temp_2": "Confusion got better after treatment, so the patient was discharged."
        }
    ]

    data = pd.read_csv('summaries_20251211_0307.csv')

    summary_temp = "discharge_summ_01"
    summary_database = pd.DataFrame(data)
    summarizer = Summarizer(model="gemini")

    i = 0
    for _, case in summary_database.iterrows():


        print(i); i+=1
        case_final_claims = pd.DataFrame()  # FIX
        try:
            summary_text = case[summary_temp]
            claims = summarizer.decompose(summary_text, as_list=True)
            judged_claims = summarizer.LLM_judge(claims, case["discharge_text"], case["context_textual"])

            for claim_idx, claim in enumerate(judged_claims):
                final_claim_row = {
                    "claim_id": f"{case['note_id']}_{claim_idx}",
                    "note_id": case["note_id"],
                    "summary": summary_text,
                    "discharge_text": case["discharge_text"],
                    "context_textual": case["context_textual"],
                    "claim": claim["claim"],
                    "ai_label": claim["label"],
                    "ai_justification": claim["motivation"],
                }

                case_final_claims = pd.concat(
                    [case_final_claims, pd.DataFrame([final_claim_row])],
                    ignore_index=True
                )

        except Exception as e:
            print("\nError processing case:")
            print(f"  note_id: {case['note_id']}")
            print(f"  summary text: {case[summary_temp]}")
            print(f"  error: {str(e)}\n")
            continue

        write_header = not os.path.isfile(save_path)
        case_final_claims.to_csv(save_path, mode='a', header=write_header, index=False)

        claims = None
        judged_claims = None
        gc.collect()
