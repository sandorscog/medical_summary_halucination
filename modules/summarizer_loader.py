from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import google.generativeai as genai
import json

def load_bart():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

def load_bart_pipeline():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return pipeline("summarization", model=model, tokenizer=tokenizer)

def load_clinical_t5():
    tokenizer = AutoTokenizer.from_pretrained("hossboll/clinical-t5")
    model = AutoModelForSeq2SeqLM.from_pretrained("hossboll/clinical-t5")
    return tokenizer, model

def load_clinical_t5_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("hossboll/clinical-t5")
    model = AutoModelForSeq2SeqLM.from_pretrained("hossboll/clinical-t5")
    return pipeline("summarization", model=model, tokenizer=tokenizer)

def load_gemini(model: str='gemini-2.5-flash'):

    with open('config/cred.json') as f:
        config = json.load(f)
    api_key = config['api_key']

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model)
    return model
