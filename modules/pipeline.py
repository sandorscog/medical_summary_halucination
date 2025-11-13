import pandas as pd
from transformers import pipeline
import evaluate
from datetime import datetime
import os

from modules.summarizer import Summarizer
from modules.data_handler import load_discharges, load_patient_data
from modules.context_handler import context_builder_text, context_builder_json, build_discharge_windows, get_patient_window_data
from modules.summarizer_loader import load_bart, load_clinical_t5, load_gemini



def data_preparation():
    """
    Loads discharges, patient data, builds contextual windows inline,
    and prepares tuples of (subject_id, hadm_id, discharge_text, context_text).
    """
    discharges = load_discharges().sort_values(['subject_id', 'storetime']).copy()
    patients_data = load_patient_data()

    # Compute time windows inline
    discharges['window_start'] = discharges.groupby('subject_id')['storetime'].shift(1)
    discharges['window_start'] = discharges['window_start'].fillna(pd.Timestamp.min)
    discharges['window_end'] = discharges['storetime']

    cases_data = []

    # Group once per subject to reduce repeated lookups
    for subject_id, group in discharges.groupby('subject_id'):
        for case in group.itertuples(index=False):
            hadm_id = case.hadm_id
            original_text = case.text

            # Select time window directly from the same row
            window_start, window_end = case.window_start, case.window_end

            # Filter patient data within that time window
            context_df = patients_data[
                (patients_data['subject_id'] == subject_id)
                & (patients_data['starttime'] >= window_start)
                & (patients_data['starttime'] <= window_end)
            ]

            # Build structured + textual context
            context_json = context_builder_json(context_df)
            context_textual = context_builder_text(context_json)

            cases_data.append(
                (subject_id, hadm_id, original_text, context_textual)
            )

    cases_data = pd.DataFrame(cases_data)
    return cases_data


def judge(claims: list[str], discharge_text: str, context_json: dict) -> list[dict]:

    judged_claims = []
    for claim in claims:

        judged_claims.append({
            'claim': claim,
            'label': 'entailment REMOVE ME',
            'score': 0, ## REMOVE THIS
        })

    return judged_claims


def claim_corretor(claims: list[dict], discharge_text: str, context_json: dict) -> list[str]:

    kept_claims = []
    for claim in claims:
        if 'contradiction' in claim['label']:
            continue
        kept_claims.append(claim)

    return kept_claims


def results_eval(original: str, summary: str, enhanced_summary: str) -> dict:
    """Compute ROUGE metrics for base and enhanced summaries."""
    rouge = evaluate.load('rouge')
    scores = rouge.compute(
        predictions=[summary, enhanced_summary],
        references=[original, original],
        use_aggregator=False
    )

    return {
        "base_rouge1": scores["rouge1"][0],
        "base_rouge2": scores["rouge2"][0],
        "base_rougeL": scores["rougeL"][0],
        "base_rougeLsum": scores["rougeLsum"][0],
        "enhanced_rouge1": scores["rouge1"][1],
        "enhanced_rouge2": scores["rouge2"][1],
        "enhanced_rougeL": scores["rougeL"][1],
        "enhanced_rougeLsum": scores["rougeLsum"][1],
    }


def append_result_to_csv(result_dict: dict, path: str):
    """Append a single result dict to a CSV file, creating it if it doesnt exist."""
    df = pd.DataFrame([result_dict])

    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)


def complete_pipeline(save_path: str = "results.csv"):
    """Run full summarization - decomposition - judging - evaluation pipeline."""
    data = data_preparation()
    summarizer = Summarizer(model='gemini')

    for i, case in enumerate(data.itertuples(index=False), start=1):
        subject_id = getattr(case, "subject_id", None)
        hadm_id = getattr(case, "hadm_id", None)
        discharge_text = case.discharge_text
        context_json = case.context_json

        print(f"\nProcessing case {i} - subject {subject_id}, hadm {hadm_id}\n Starting summary...")

        try:
            summary = summarizer.summ(discharge_text)
            print('Decomposing into claims...')
            claims = summarizer.decompose(summary, as_list=True)

            print(f'Processing {len(claims)} claims...')
            judged_claims = judge(claims, discharge_text, context_json)
            final_claims = claim_corretor(judged_claims, discharge_text, context_json)

            print('Compiling new summary')
            enhanced_summary = summarizer.compile(final_claims)

            metrics = results_eval(discharge_text, summary, enhanced_summary)

            # --- save partial result immediately ---
            result = {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "original_text": discharge_text,
                "context_json": context_json,
                "base_summary": summary,
                "decomposed_claims": "\n".join(claims),
                "judged_claims": "\n".join(judged_claims),
                "final_claims": "\n".join(final_claims),
                "enhanced_summary": enhanced_summary,
                **metrics,
            }

            append_result_to_csv(result, save_path)
            print(f"Case {i} saved to {save_path}")

        except Exception as e:
            print(f"ERROR!! on case {i}: {e}")
            continue

    # --- rename final file with timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = f"results_{timestamp}.csv"
    os.rename(save_path, final_path)
    print(f"\n!! Pipeline complete !! Results saved as {final_path}")

