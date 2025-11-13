from modules.summarizer_loader import load_bart, load_clinical_t5, load_gemini
import evaluate
import pandas as pd


class Summarizer():
    def __init__(self, model: str = 'clinical', config: dict={}):
        if model == 'bart':
            self.tokenizer, self.model = load_bart()
        elif model == 'clinical':
            self.tokenizer, self.model = load_clinical_t5()
        elif model == 'gemini':
            self.model = load_gemini()
            self.tokenizer = None


    def summ(self, text: str, prompt: str="Summarize the following discharge note. Don't include visual elements in the text: \n"):

        if self.tokenizer:
            tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
            output_tokens = self.model.generate(tokens, max_length=256, num_beams=4, early_stopping=True)
            output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        else:
            output = self.model.generate_content(prompt + text).text

        return output
    
    def decompose(self, summary_text: str, as_list: bool = False):

        DECOMPOSITION_PROMPT_TEMPLATE = '''

            You are a medical information analyst. Your task is to parse the entire medical summary below and decompose it into a complete list of fine-grained, verifiable claims. These claims will be used to fact-check the summary against a full patient record.

            Each claim must be a single, self-contained medical fact, finding, instruction, or patient-reported statement.

            **Extraction Rules:**
            1.  **Split Lists:** Decompose lists of conditions, symptoms, findings, or plans into separate claims (e.g., "history of A, B, and C" becomes three separate claims).
            2.  **Keep Instructions Intact:** A single medication (e.g., "Furosemide 40mg daily") is one complete claim. Do not split the drug from its dosage.
            3.  **Be Self-Contained:** Resolve all pronouns (e.g., "She," "He") to "The patient."
            4.  **Extract All Details:** Capture demographics, past medical history (PMH), symptoms, lab results, imaging findings, diagnoses, medications, and follow-up plans.
            5.  **Output Format:** Provide the output as a simple list of claims one in each line.

            ---
            ### Example 1

            **Full Summary:**
            The patient, a female with a history of HCV cirrhosis with ascites, HIV on ART, IV drug use, COPD, bipolar disorder, and PTSD, presented with worsening abdominal distension and pain. She had discontinued her Lasix and Spironolactone prior to admission due to feeling that they weren't effective and wanting to avoid "chemicals."During hospitalization, the patient's ascites was managed with diuretics (Furosemide 40mg and Spironolactone 50mg daily). Paracentesis was attempted in the ED but was unsuccessful. Lab results showed elevated liver enzymes (ALT, AST, Alk Phos, Total Bili), and a low platelet count. A CXR showed no acute cardiopulmonary process. Ultrasound revealed liver nodularity, signs of portal hypertension (ascites and splenomegaly), and cholelithiasis. The patient was scheduled for follow-up with her current PCP, a new PCP (Dr. ___), and the Liver Clinic for EGD and ___. She was discharged home in a clear and coherent mental state, alert, interactive, and ambulatory, with a diagnosis of ascites from portal hypertension. She was instructed to continue taking Furosemide 40mg and Spironolactone 50mg daily, adhere to a low-salt diet, and follow up with the Liver Clinic.


            **Claims:**
            The patient is female.

            The patient has a history of HCV cirrhosis with ascites.

            The patient has a history of HIV on ART.

            The patient has a history of IV drug use.

            The patient has a history of COPD.

            The patient has a history of bipolar disorder.

            The patient has a history of PTSD.

            The patient presented with worsening abdominal distension.

            The patient presented with worsening abdominal pain.

            The patient had discontinued her Lasix prior to admission.

            The patient had discontinued her Spironolactone prior to admission.

            The patient's reason for discontinuing was feeling [the drugs] weren't effective.

            The patient's reason for discontinuing was wanting to avoid "chemicals."

            The patient's ascites was managed with Furosemide 40mg daily during hospitalization.

            The patient's ascites was managed with Spironolactone 50mg daily during hospitalization.

            Paracentesis was attempted in the ED.

            The paracentesis attempt in the ED was unsuccessful.

            Lab results showed elevated liver enzymes (ALT, AST, Alk Phos, Total Bili).

            Lab results showed a low platelet count.

            A CXR showed no acute cardiopulmonary process.

            Ultrasound revealed liver nodularity.

            Ultrasound revealed signs of portal hypertension (ascites and splenomegaly).

            Ultrasound revealed cholelithiasis.

            The patient was scheduled for follow-up with her current PCP.

            The patient was scheduled for follow-up with a new PCP (Dr. ___).

            The patient was scheduled for follow-up with the Liver Clinic for EGD.

            The patient was scheduled for follow-up with the Liver Clinic for ___.

            The patient was discharged home.

            The patient's mental state at discharge was clear and coherent.

            The patient's status at discharge was alert.

            The patient's status at discharge was interactive.

            The patient's status at discharge was ambulatory.

            The patient's diagnosis was ascites from portal hypertension.

            The patient was instructed to continue taking Furosemide 40mg daily.

            The patient was instructed to continue taking Spironolactone 50mg daily.

            The patient was instructed to adhere to a low-salt diet.

            The patient was instructed to follow up with the Liver Clinic.

            ---
            ### New summary (YOUR TASK)

            **Full Summary:**
            {summary_text}
        '''

        prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(summary_text=summary_text)
        response = self.model.generate_content(prompt).text.strip()

        if as_list:
            claims = [line.strip() for line in response.splitlines() if line.strip()]
            return claims
        else:
            return response
    
    def compile(self, claims: list[str]) -> str:

        prompt = '''
            You are a precise and concise medical writer.
            Your task is to merge and rewrite the following factual sentences into a single, well-structured text.

            Guidelines:
             - Only combine sentences if their meanings are fully equivalent or trivially overlapping.
             - Maintain all factual information — do not omit or change any details.
             - If there is any uncertainty about whether two sentences mean the same thing, keep both.
             - Preserve the order of the claims
             - Use a natural narrative tone, combining ideas logically (e.g., history -> reason for admission -> findings).
             - If multiple facts are related, integrate them into one flowing sentence.
             - Preserve medical accuracy, structure, and readability.
             - Do not speculate or interpret beyond what is written.


            Example 1:
            Input:
            The patient reported self-discontinuing her diuretics.
            The patient reported self-discontinuing Lasix.
            The patient reported self-discontinuing Spironolactone.
            The patient reported not adhering to a sodium-restricted diet.
            Self-discontinuing diuretics and not adhering to a sodium-restricted diet likely contributed to the worsening ascites.


            Output:
            The patient reported self-discontinuing her diuretics, including Lasix and Spironolactone, and not adhering to a sodium-restricted diet, both of which likely contributed to her worsening ascites.


            Example 2:
            Input:
            The patient has a history of HCV cirrhosis.
            The patient has a history of HIV on ART.
            The patient has a history of COPD.
            The patient has a history of bipolar disorder.
            The patient has a history of PTSD.

            Output:
            The patient has a history of HCV cirrhosis, HIV on ART, COPD, bipolar disorder, and PTSD.

            *NOW AGGREGATE (YOUR TASK)*:
            

        '''

        claims_text = "\n".join(claims)
        prompt += "\n" + claims_text.strip()
        return self.model.generate_content(prompt).text.strip()



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


