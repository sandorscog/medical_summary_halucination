from warnings import warn
import evaluate
import pandas as pd
from transformers import BartTokenizer, AutoTokenizer
from modules.summarizer_loader import load_bart_pipeline, load_clinical_t5_pipeline, load_bart, load_clinical_t5

class EvalSummaryManager():

    def __init__(self):
        self.models = []
        self.bertscore = evaluate.load('bertscore')
        self.rouge = evaluate.load('rouge')

    def add_model(self, model_name: str):
        if model_name == 'bart':
            tokenizer, _ = load_bart()
            self.models.append({
                'model_name': 'bart',
                'model': load_bart_pipeline(),
                'tokenizer': tokenizer,
                'max_len': tokenizer.model_max_length-3,  # usually 1024
            })
        elif model_name == 't5':
            tokenizer, _ = load_clinical_t5()
            self.models.append({
                'model_name': 't5',
                'model': load_clinical_t5_pipeline(),
                'tokenizer': tokenizer,
                'max_len': tokenizer.model_max_length-3,  # usually 512
            })
        else:
            warn('No model with the given name!! Nothing was loaded')

    def truncate_by_tokens(self, text: str, tokenizer, max_tokens: int) -> str:
        # Tokenize with truncation to max_tokens, return decoded string without special tokens
        encoded = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors='pt')
        return tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)

    def everything_eval(self, texts: list):
        results = []

        for text in texts:
            for model in self.models:
                # Truncate text according to the model's max token length
                short_text = self.truncate_by_tokens(text, model['tokenizer'], model['max_len'])

                # Generate summary from truncated text
                row = model['model'](short_text)[0]
                summary = row['summary_text']

                # Evaluate summary with bertscore and rouge against original full text
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
