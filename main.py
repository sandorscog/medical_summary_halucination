from modules.summarizer import Summarizer
from modules.SummaryManager import EvalSummaryManager
from modules.summary_enhancer import enhance
from modules.data_handler import load_data



def main():

    data = load_data()
    
    summarizer = Summarizer(model='bart')
    evaluator = EvalSummaryManager()
    for case in data:

        original_text = case['discharge']
        base_summary = summarizer.summ(original_text)
        base_evaluation = evaluator.eval_summary(summary=base_summary, original=original_text)

        enhanced_summary = enhance(base_summary, case)
        enhanced_evaluation = evaluator.eval_summary(summary=enhanced_summary, original=original_text)




if __name__ == '__main__':
    main()