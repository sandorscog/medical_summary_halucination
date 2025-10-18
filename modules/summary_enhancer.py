


def claims_parcer(summary: str) -> list:
    return summary.split('.')


def claim_judge(claims) -> list:

    judged_claims = [(str, bool)]
    for claim in claims:
        ## TODO: Define ways to tell if a statement is true, false or incomplete
        judged_claims.append((claim, False)) ### REMOVE THIS LINE

    return judged_claims


def enhance(base_summary: str, context):


    claims = claims_parcer(base_summary)

    improved_summary = base_summary

    return improved_summary