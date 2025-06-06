from modules.summarizer_loader import load_bart, load_clinical_t5
import evaluate
import pandas as pd


class Summarizer():
    def __init__(self, model: str = 'clinical'):
        if model == 'bart':
            self.tokenizer, self.model = load_bart()
        else:
            self.tokenizer, self.model = load_clinical_t5()


    def summ(self, text: str):
        tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        output_tokens = self.model.generate(tokens, max_length=256, num_beams=4, early_stopping=True)
        output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return output
    
    def rogue(summ='', reference=''):
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=[summ], references=[reference])
        return results
    
    def summ_rogue(summ='', reference=''):
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=[summ], references=[reference])
        results["reference"] = reference
        results["generated"] = summ
        return results


