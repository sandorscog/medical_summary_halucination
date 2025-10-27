

def correct_false_claim(false_claim, context):
    return false_claim


def correct_partial_claims(partial_claim, context):
    return partial_claim


def text_compiler(claims):
    compiled_text = ''
    for claim in claims:
        compiled_text += claim
    return compiled_text


def corrector(claims, context):

    corrected_claims = []
    for claim in claims:
        if claim[1] == 'False':
            corrected_claims.append(correct_false_claim(claim[0]), context)
        elif claim[1] == 'Parcial':
            corrected_claims.append(correct_partial_claims(claim[0]), context)
        else:
            corrected_claims.append(claim[0])



def claims_parcer(summary: str) -> list:
    return summary.split('.')


def claim_judge(claims) -> list:

    judged_claims = [(str, str)]
    for claim in claims:
        ## TODO: Define ways to tell if a statement is true, false or incomplete
        judged_claims.append((claim, 'False')) ### REMOVE THIS LINE

    return judged_claims


def enhancer(base_summary: str, context):
    claims = claims_parcer(base_summary)
    judged_claims = claim_judge(claims, context)
    corrected_claims = corrector(judged_claims, context)
    improved_summary = text_compiler(corrected_claims)

    return improved_summary