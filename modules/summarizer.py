from modules.summarizer_loader import load_bart, load_clinical_t5, load_gemini
import evaluate
import pandas as pd
from transformers import pipeline
import re
from collections import Counter


class Summarizer():
    def __init__(self, model: str = 'clinical', config: dict={}, temp=1):
        if model == 'bart':
            self.tokenizer, self.model = load_bart()
        elif model == 'clinical':
            self.tokenizer, self.model = load_clinical_t5()
        elif model == 'gemini':
            self.model = load_gemini(temp=temp)
            self.tokenizer = None

        self.nli = pipeline("text-classification", model="facebook/bart-large-mnli")


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

            Extraction Rules:
            1.  Split Lists: Decompose lists of conditions, symptoms, findings, or plans into separate claims (e.g., "history of A, B, and C" becomes three separate claims).
            2.  Keep Instructions Intact: A single medication (e.g., "Furosemide 40mg daily") is one complete claim. Do not split the drug from its dosage.
            3.  Be Self-Contained: Resolve all pronouns (e.g., "She," "He") to "The patient."
            4.  Extract All Details: Capture demographics, past medical history (PMH), symptoms, lab results, imaging findings, diagnoses, medications, and follow-up plans.
            5.  Output Format: Provide the output as a simple list of claims one in each line.

            ---
            ### Example 1

            Full Summary:
            The patient, a female with a history of HCV cirrhosis with ascites, HIV on ART, IV drug use, COPD, bipolar disorder, and PTSD, presented with worsening abdominal distension and pain. She had discontinued her Lasix and Spironolactone prior to admission due to feeling that they weren't effective and wanting to avoid "chemicals."During hospitalization, the patient's ascites was managed with diuretics (Furosemide 40mg and Spironolactone 50mg daily). Paracentesis was attempted in the ED but was unsuccessful. Lab results showed elevated liver enzymes (ALT, AST, Alk Phos, Total Bili), and a low platelet count. A CXR showed no acute cardiopulmonary process. Ultrasound revealed liver nodularity, signs of portal hypertension (ascites and splenomegaly), and cholelithiasis. The patient was scheduled for follow-up with her current PCP, a new PCP (Dr. ___), and the Liver Clinic for EGD and ___. She was discharged home in a clear and coherent mental state, alert, interactive, and ambulatory, with a diagnosis of ascites from portal hypertension. She was instructed to continue taking Furosemide 40mg and Spironolactone 50mg daily, adhere to a low-salt diet, and follow up with the Liver Clinic.


            Claims:
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

            Full Summary:
            {summary_text}
        '''

        prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(summary_text=summary_text)
        response = self.model.generate_content(prompt).text.strip()

        if as_list:
            claims = [line.strip() for line in response.splitlines() if line.strip()]
            return claims
        else:
            return response
        

    def judge(self, claims: list[str], discharge_text: str, context_text: str) -> list[dict]:
        """
        Run NLI on each claim against the discharge summary + context_json.
        Chunking is applied to avoid model context limits.
        Returns: list of dict {claim, label, score}
        """

        premise = (
            f"Discharge:\n{discharge_text}\n\n"
            f"Patient context:\n{context_text}"
        )

        chunk_size = 2000
        overlap = 200

        premise_chunks = [
            premise[i : i + chunk_size]
            for i in range(0, len(premise), chunk_size - overlap)
        ]

        judged = []
        for claim in claims:
            best_label = None
            best_score = 0.0

            # Evaluate the claim against each chunk
            for chunk in premise_chunks:

                # your original input format: chunk </s> claim
                result = self.nli(f"{chunk} </s> {claim}")
                label  = result[0]["label"]
                score  = result[0]["score"]

                # keep the strongest chunk
                if score > best_score:
                    best_label = label
                    best_score = score

                # early stopping if confident entailment
                if label == "ENTAILMENT" and score > 0.75:
                    best_score = score
                    break

            judged.append({
                "claim": claim,
                "label": best_label,
                "score": float(best_score),
            })

        return judged

    
    def LLM_judge(self, claims: list[str], discharge_text: str, context_text: str):

        judged = []
        premise = (
            f"Discharge:\n{discharge_text}\n\n"
            f"Patient context:\n{context_text}"
        )

        prompt_template = '''
        
            You are a medical analyst and fact-checker. Your job is to judge clinical claims about a specific patient by carefully reading only the text provided (discharge note and related context sections). You must not use any outside knowledge or invent facts beyond the provided text. Focus strictly on whether the CLAIM is supported, contradicted, or cannot be determined from the supplied text. Your output must be strictly evidence-grounded and follow the exact output format described below.

            INSTRUCTIONS (read carefully):
            1) You are given: (A) a discharge note and other context sections from the patient's chart, (B) a single CLAIM to judge.
            2) Use ONLY the provided text as evidence. Do NOT use external knowledge or make assumptions that are not supported by the text.
            3) Look for explicit statements and strong implications in the text that support or contradict the claim. If the text only *lists medications* in a discharge-medications section, treat that as a strong implication of continuation unless there is explicit contradictory language (e.g., "discontinued", "stopped", "hold", "do not resume") in the discharge or plan.
            4) For every judgment you make, quote the exact sentence(s) or contiguous text span(s) from the provided text that you used as evidence. Put each quoted piece on its own line (a quoted block per evidence piece).
            5) If more than one separate section supports/contradicts the claim, include each supporting/contradicting quote on separate lines in the order you used them.
            6) After the quoted evidence lines, output a single line with one token LABEL which must be exactly one of:
            - ENTAILMENT
            - CONTRADICTION
            - NEUTRAL
            7) Do not output anything else — no preamble, no extra commentary, and no hallucinated citations.

            ------------------------------------------------------------
            FEW-SHOT EXAMPLES
            ------------------------------------------------------------

            ### EXAMPLE 1
            CLAIM:
            "The patient was advised to continue her HIV medications at discharge."

            EVIDENCE:
            "Discharge Medications: ... 2. Emtricitabine-Tenofovir (Truvada) 1 TAB PO DAILY ... 6. Raltegravir 400 mg PO BID"
            "Dear Ms. ___, ... Please take these medications daily to keep excess fluid off ..."
            ENTAILMENT

            ---

            ### EXAMPLE 2
            CLAIM:
            "The patient was confused at the time of discharge."

            EVIDENCE:
            "Discharge Condition: Mental Status: Clear and coherent. Level of Consciousness: Alert and interactive."
            CONTRADICTION

            ---

            ### EXAMPLE 3
            CLAIM:
            "A successful paracentesis was performed during the hospitalization."

            EVIDENCE:
            "Diagnostic para attempted in the ED, unsuccessful."
            "Discharge Instructions: ... we did a paracentesis to remove 1.5L of fluid from your belly."
            ENTAILMENT

            ------------------------------------------------------------
            END OF EXAMPLES
            ------------------------------------------------------------


            --- 
            INPUT:
            [DISCHARGE NOTE AND CONTEXT SECTIONS — keep section headers]
            {premise}

            CLAIM:
            {claim}
        '''

        for claim in claims:

            prompt = prompt_template.format(
                premise=premise,
                claim=claim
            )

            response = self.model.generate_content(prompt).text
            label = re.search(r'(ENTAILMENT|CONTRADICTION|NEUTRAL)', response).group(1)

            judged.append({
                "claim": claim,
                "label": label,
                "score": float(1),
                'motivation': response,
            })

        return judged


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


    def llm_judge_metrics(self, discharge_note, base_summary, new_summary) -> dict:

        prompt = '''

        You are a medical analyst and factuality evaluator. Your task is to compare two summaries (A and B) of the same discharge note and decide which one is more factually accurate.

        RULES:
        1. Use ONLY the discharge note as evidence. Do NOT use outside medical knowledge or assumptions.
        2. Judge strictly factual accuracy: correctness of events, diagnoses, medications, procedures, findings, and instructions.
        3. Penalize hallucinations (claims not present), contradictions, incorrect details, or invented clinical facts.
        4. Reward summaries that stay faithful to the discharge note, reflect the actual clinical course, and avoid speculation.
        5. You must be completely impartial. Avoid any positional or ordering bias.
        The order in which Summary A and Summary B appear must NOT influence your judgment.
        6. Read both summaries fully before making any judgment. Evaluate them independently, then compare.

        REQUIRED OUTPUT FORMAT (very simple):
        1. A short explanation of why one summary is more accurate showing the rationale behind your decision.
        2. A final line containing EXACTLY one of:
        - [A]
        - [B]

        Do NOT output anything else.
        ------------------------------------------------------------
        DISCHARGE NOTE:
        {discharge_note}
        [END OF DISCHARGE NOTE]

        SUMMARY A:
        {A}
        [END OF SUMMARY A]

        SUMMARY B:
        {B}
        [END OF SUMMARY B]



        '''


        rounds = 4
        votes = []
        for _ in range(rounds):

            # ---- Run with A first, B second ----
            prompt1 = prompt.format(
                discharge_note=discharge_note,
                A=base_summary,
                B=new_summary
            )
            response1 = self.model.generate_content(prompt1).text
            result1 = re.search(r'\[(A|B)\]', response1)
            if result1:
                votes.append(result1.group(1))

            # ---- Run with B first, A second ----
            prompt2 = prompt.format(
                discharge_note=discharge_note,
                A=new_summary,
                B=base_summary
            )
            response2 = self.model.generate_content(prompt2).text
            result2 = re.search(r'\[(A|B)\]', response2)
            if result2:
                # reverse the meaning because summaries were swapped
                votes.append("A" if result2.group(1) == "B" else "B")

        # ----- Majority vote -----
        count = Counter(votes)
        A_votes = count["A"]
        B_votes = count["B"]

        if A_votes > B_votes:
            comparative_winner = "base"
        elif B_votes > A_votes:
            comparative_winner = "enhanced"
        else:
            comparative_winner = "uncertain"

        prompt_factual = '''

        You are a medical analyst and factuality rater. Judge how faithful and high-quality the SUMMARY is relative to the DISCHARGE NOTE. Use ONLY the information present in the discharge note; do NOT use outside knowledge.

        Your evaluation should consider four dimensions:
        1. Factual Accuracy & Support  
        - Identify the important factual claims in the summary.  
        - Check whether each claim is explicitly supported, contradicted, or not supported by the discharge note.  
        - Hallucinations, contradictions, and clinically meaningful omissions must reduce the score.

        2. Faithfulness & Coverage  
        - The summary should capture the key clinical information from the discharge (diagnoses, major procedures, key clinical events, final disposition, medications when central).  
        - Penalize missing critical facts or distortions of the clinical story.

        3. Cohesion, Clarity & Structure  
        - The summary should be coherent, well-structured, and easy to follow.  
        - Disorganized, repetitive, or confusing writing should reduce the score.

        4. Brevity & Relevance  
        - Penalize unnecessary length or filler content that does not contribute meaningful or verifiable information.  
        - Overly long summaries with little added value should score lower.

        SCORING RUBRIC (0-10):
        - 9-10 (Excellent): Highly faithful to the discharge note; no major errors; at most tiny or debatable issues; cohesive and concise.  
        - 6-8 (Good): Mostly faithful; minor issues or small omissions; few or no hallucinations; generally coherent.  
        - 3-5 (Weak): Noticeable inaccuracies, hallucinations, missing key facts, or structural problems; meaningfully deviates from the discharge.  
        - 0-2 (Poor): Serious contradictions, major hallucinations, or incoherent writing; misrepresents large parts of the discharge.

        OUTPUT FORMAT (strict):
        1. A short rationale summarizing the major strengths/weaknesses.  
        2. On the next line output an integer score 0-10 inside double brackets, e.g.:  
        `[[7]]`

        Do not output anything else.

        Now evaluate:

        DISCHARGE NOTE:
        {discharge_note}
        [END]

        SUMMARY:
        {summary}
        [END]


        '''

        def evaluate(summary_text: str):
            prompt = prompt_factual.format(
                discharge_note=discharge_note,
                summary=summary_text
            )

            response = self.model.generate_content(prompt).text

            # Extract score [[X]]
            match = re.search(r"\[\[(\d{1,2})\]\]", response)
            score = int(match.group(1)) if match else None

            # Rationale = everything before [[score]]
            rationale = response[:match.start()].strip() if match else response.strip()

            return rationale, score

        # Evaluate Base Summary
        base_rationale, base_score = evaluate(base_summary)

        # Evaluate New Summary
        new_rationale, new_score = evaluate(new_summary)

        return {
            'best_summary': comparative_winner,
            'base_llm_score': base_score,
            'enhanced_llm_score': new_score,
        }




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

