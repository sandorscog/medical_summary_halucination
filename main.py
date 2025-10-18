import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from modules.summarizer import Summarizer
from modules.SummaryManager import EvalSummaryManager
from modules.summary_enhancer import enhance
from modules.data_handler import load_discharges, load_patient_data
from modules.context_handler import context_builder_text, context_builder_json, build_discharge_windows, get_patient_window_data

import pandas as pd




def main():
    discharges = load_discharges()          # DataFrame with discharge info
    patients_data = load_patient_data()     # DataFrame with all patient records

    # Precompute discharge windows once
    discharge_windows = build_discharge_windows(discharges)
    print(discharge_windows)

    summarizer = Summarizer(model='bart')
    evaluator = EvalSummaryManager()

    for _, case in discharges.iterrows():
        subject_id = case['subject_id']
        hadm_id = case['hadm_id']
        original_text = case['text']

        # Retrieve contextual data for that discharge
        context_df = get_patient_window_data(
            subject_id=subject_id,
            hadm_id=hadm_id,
            windows_dict=discharge_windows,
            patient_df=patients_data,
            time_col='starttime'
        )

        # Convert to text/JSON context for the LLM
        context_textual = context_builder_text(context_df)
        context_json = context_builder_json(context_df)

        # Run summarization and evaluation
        base_summary = summarizer.summ(original_text)
        base_evaluation = evaluator.eval_summary(summary=base_summary, original=original_text)

        enhanced_summary = enhance(base_summary, context_textual)
        enhanced_evaluation = evaluator.eval_summary(summary=enhanced_summary, original=original_text)



if __name__ == '__main__':
    main()