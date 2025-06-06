from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

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