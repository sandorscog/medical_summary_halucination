from warnings import warn
import evaluate
import pandas as pd
from transformers import BartTokenizer, AutoTokenizer
from modules.summarizer_loader import load_bart_pipeline, load_clinical_t5_pipeline, load_bart, load_clinical_t5, load_gemini

class EvalSummaryManager():

    def __init__(self):
        self.models = []
        self.bertscore = evaluate.load('bertscore')
        self.rouge = evaluate.load('rouge')

    def eval_summary(self, summary: str, original: str):

        bert_score = self.bertscore.compute(predictions=[summary], references=[original], lang='en')
        rouge_score = self.rouge.compute(predictions=[summary], references=[original])

        return {
            'bert-score': bert_score,
            'rouge': rouge_score,
        }

    def add_model(self, model_name: str):
        if model_name == 'bart':
            tokenizer, model = load_bart()
            self.models.append({
                'model_name': 'bart',
                'model': model,
                'tokenizer': tokenizer,
                'max_len': tokenizer.model_max_length-1,
            })
        elif model_name == 't5':
            tokenizer, model = load_clinical_t5()
            self.models.append({
                'model_name': 't5',
                'model': model,
                'tokenizer': tokenizer,
                'max_len': tokenizer.model_max_length-1,
            })
        elif model_name == 'gemini':
            model = load_gemini()
            self.models.append({
                'model_name': 'gemini',
                'model': model,
                'tokenizer': None,
                'max_len': None,
            })

        else:
            warn('No model with the given name!! Nothing was loaded')

    def truncate_by_tokens(self, text: str, tokenizer, max_tokens: int) -> str:
        # Tokenize with truncation to max_tokens, return decoded string without special tokens
        encoded = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors='pt')
        return tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)

    def everything_eval(self, texts: list, prompt: str='summarize the following text: \n'):
        results = []

        for text in texts:
            for model in self.models:

                if model['model_name'] == 'bart' or model['model_name'] == 't5':
                    # Truncate text
                    short_text = self.truncate_by_tokens(text, model['tokenizer'], model['max_len'])

                    # Encode inputs properly
                    encoded = model['tokenizer'](
                        short_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=model['max_len']
                    )

                    # Generate summary using .generate()
                    summary_ids = model['model'].generate(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        max_length=256,
                        num_beams=4,
                        early_stopping=True
                    )
                    summary = model['tokenizer'].decode(summary_ids[0], skip_special_tokens=True)

                elif model['model_name'] == 'gemini':
                    summary = model['model'].generate_content(prompt + text).text


                # Evaluate with BERTScore and ROUGE
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

