from warnings import warn
import evaluate
import pandas as pd

from modules.summarizer_loader import load_bart_pipeline, load_clinical_t5_pipeline

class EvalSummaryManager():

    def __init__(self):
        self.models = []
        self.bertscore = evaluate.load('bertscore')
        self.rouge = evaluate.load('rouge')

    def add_model(self, model_name: str):
        if model_name == 'bart':
            self.models.append({
                'model_name': 'bart',
                'model': load_bart_pipeline(),
            })
        elif model_name == 't5':
            self.models.append({
                'model_name': 't5',
                'model': load_clinical_t5_pipeline(),
            })
        else:
            warn('No model with the given name!! Nothing was loaded')

    def everything_eval(self, texts: list):

        results = []

        for text in texts:
            for model in self.models:
                row = model['model'](text)[0]
                summary = row['summary_text']

                bert = self.bertscore.compute(predictions=[summary], references=[text], lang='en')
                rouge = self.rouge.compute(predictions=[summary], references=[text])

                result_row = {
                    'text': text,
                    'model': model['model_name'],
                    'summary': summary,
                    'bert_precision': bert['precision'][0],
                    'bert_recall': bert['recall'][0],
                    'bert_f1': bert['f1'][0],
                    'rouge1': rouge.get('rouge1', 0.0),
                    'rouge2': rouge.get('rouge2', 0.0),
                    'rougeL': rouge.get('rougeL', 0.0)
                }

                results.append(result_row)

        return pd.DataFrame(results)
                