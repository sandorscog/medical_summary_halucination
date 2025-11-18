import pandas as pd
from transformers import pipeline
import evaluate
from datetime import datetime
import os
import traceback

from modules.summarizer import Summarizer
from modules.data_handler import load_discharges, load_patient_data
from modules.context_handler import context_builder_text, context_builder_json, build_discharge_windows, get_patient_window_data
from modules.summarizer_loader import load_bart, load_clinical_t5, load_gemini



def data_preparation():
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
            context_df = patients_data[
                (patients_data['subject_id'] == subject_id)
                & (patients_data['starttime'] >= window_start)
                & (patients_data['starttime'] <= window_end)
            ]

            # -----------------------------------------
            # Build JSON + text versions of the context
            # -----------------------------------------
            context_json = context_builder_json(context_df)
            context_textual = context_builder_text(context_json)

            cases_data.append(
                (subject_id, hadm_id, original_text, context_json, context_textual)
            )

    # -------------------------------
    # Convert to DataFrame and return
    # -------------------------------
    cases_data = pd.DataFrame(
        cases_data,
        columns=["subject_id", "hadm_id", "discharge_text", "context_json", "context_textual"]
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
        if 'contradiction' in claim['label']:
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
        "base_rouge2": rouge_scores["rouge2"][0],
        "base_rougeL": rouge_scores["rougeL"][0],
        "base_rougeLsum": rouge_scores["rougeLsum"][0],

        # ---- ROUGE ENHANCED ----
        "enhanced_rouge1": rouge_scores["rouge1"][1],
        "enhanced_rouge2": rouge_scores["rouge2"][1],
        "enhanced_rougeL": rouge_scores["rougeL"][1],
        "enhanced_rougeLsum": rouge_scores["rougeLsum"][1],

        # ---- BERTScore BASE ----
        "base_bertscore_precision": bert_scores["precision"][0],
        "base_bertscore_recall": bert_scores["recall"][0],
        "base_bertscore_f1": bert_scores["f1"][0],

        # ---- BERTScore ENHANCED ----
        "enhanced_bertscore_precision": bert_scores["precision"][1],
        "enhanced_bertscore_recall": bert_scores["recall"][1],
        "enhanced_bertscore_f1": bert_scores["f1"][1],
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

    for i, case in enumerate(data.iloc[13:].itertuples(index=False), start=1):
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
            judged_claims = summarizer.judge(claims, discharge_text, context_text) ## TEST WITH CONTEXTUAL TEXT
            final_claims = claim_corretor(judged_claims, discharge_text, context_json)

            print('Compiling new summary')
            enhanced_summary = summarizer.compile([claim['claim'] for claim in final_claims])

            metrics = results_eval(discharge_text, summary, enhanced_summary)


            result = {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "original_text": discharge_text,
                "context_json": context_json,
                "base_summary": summary,
                "decomposed_claims": "\n".join(claims),
                "judged_claims": "\n".join(f"{jc['claim']} - {jc['label']} - {jc['score']:.3f}" for jc in judged_claims),
                "final_claims": "\n".join(claim['claim'] for claim in final_claims),
                "enhanced_summary": enhanced_summary,
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

