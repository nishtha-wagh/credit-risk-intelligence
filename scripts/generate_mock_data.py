"""
generate_mock_data.py

Generates a synthetic credit risk dataset:
  - data/raw/borrowers.csv       (structured signals)
  - data/raw/case_notes.csv      (unstructured text notes)

Run:
    python scripts/generate_mock_data.py --n 200 --seed 42
"""

import argparse
import csv
import random
import uuid
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOAN_TYPES = ["auto", "mortgage", "personal", "student"]
EMPLOYMENT_STATUSES = ["employed", "self-employed", "unemployed", "retired"]
NOTE_TYPES = ["underwriter", "collections", "servicing", "complaint"]
AUTHOR_ROLES = {"underwriter": "underwriter", "collections": "analyst",
                "servicing": "agent", "complaint": "agent"}

NOTE_TEMPLATES = {
    "underwriter": [
        "Borrower presented stable employment documentation. DTI within acceptable range. No derogatory marks on credit report. Approved with standard terms.",
        "Noted employment gap of {gap} months prior to application. Borrower provided explanation letter referencing medical leave. Approved with enhanced monitoring flag.",
        "FICO score of {fico} observed. Self-employment income verified via two years of tax returns. Recommend approval with income re-verification at 12 months.",
        "Debt-to-income ratio of {dti:.0%} exceeds standard threshold of 43%. Compensating factors: strong payment history, significant cash reserves. Approved with conditions.",
        "Borrower has {late_30} late payments in last 24 months. Explanation provided — disputed billing error since resolved. Proceeding with caution; flagged for 90-day review.",
        "Application flagged by automated scoring model. Manual review completed. Risk tier assigned: {risk_tier}. Decision consistent with policy R-114.",
    ],
    "collections": [
        "Outbound contact attempt {n}. No response. Letter sent to address on file. Account placed on 30-day hold pending contact.",
        "Spoke with borrower on {date}. Borrower reports temporary hardship due to job loss. Agreed to 60-day deferral. Account status updated. Next review scheduled.",
        "Borrower has missed {late_60} consecutive payments. Standard collections process initiated. Legal referral pending if no payment arrangement by end of month.",
        "Payment arrangement of ${amount}/month agreed. First payment due in 14 days. Arrangement logged. Agent: {agent}.",
        "Account referred to external collections agency. Balance outstanding: ${balance}. Borrower unreachable for 90 days. Internal case closed.",
        "Borrower contacted via verified number. Expressed willingness to pay. Partial payment of ${amount} received. Remaining balance restructured over 6 months.",
    ],
    "servicing": [
        "Routine account review completed. Payment history satisfactory. No action required.",
        "Borrower requested payoff statement. Document issued via secure portal. Loan in good standing.",
        "Escrow analysis completed. Adjusted monthly payment by ${adj} to reflect updated insurance premium.",
        "Borrower reported change of address. Records updated. Identity verification completed per policy.",
        "Automatic payment failed — insufficient funds. Borrower notified. Manual payment window: 5 days before late fee applied.",
        "Annual review: account performing within expected parameters. Risk tier maintained at {risk_tier}.",
    ],
    "complaint": [
        "Borrower filed formal complaint regarding billing statement discrepancy. Investigation initiated. Response due within 30 days per CFPB guidelines.",
        "Borrower disputes late fee charge on {date}. Payment records reviewed — payment received within grace period. Fee reversal approved.",
        "Escalated complaint: borrower alleges unauthorised credit pull. Compliance team reviewing. Borrower notified of investigation timeline.",
        "Complaint resolved. Root cause: system error duplicated payment posting. Correction applied. Borrower confirmed satisfied with resolution.",
    ],
}

RISK_TIER_RULES = {
    "LOW": lambda b: b["fico_score"] >= 720 and b["dti_ratio"] <= 0.36 and b["num_deferrals"] == 0,
    "MEDIUM": lambda b: 650 <= b["fico_score"] < 720 or 0.36 < b["dti_ratio"] <= 0.45,
    "HIGH": lambda b: b["fico_score"] < 650 or b["dti_ratio"] > 0.45 or b["num_deferrals"] >= 2,
    "CRITICAL": lambda b: b["ever_defaulted"] or b["payments_late_90d"] >= 2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))


def assign_risk_tier(b: dict) -> str:
    if RISK_TIER_RULES["CRITICAL"](b):
        return "CRITICAL"
    if RISK_TIER_RULES["HIGH"](b):
        return "HIGH"
    if RISK_TIER_RULES["MEDIUM"](b):
        return "MEDIUM"
    return "LOW"


def fill_template(template: str, b: dict, extra: dict = None) -> str:
    ctx = {
        "fico": b.get("fico_score", 700),
        "dti": b.get("dti_ratio", 0.38),
        "gap": b.get("employment_gap_months", 0),
        "late_30": b.get("payments_late_30d", 0),
        "late_60": b.get("payments_late_60d", 0),
        "risk_tier": b.get("analyst_risk_tier", "MEDIUM"),
        "date": random_date(date(2022, 1, 1), date(2024, 6, 1)).strftime("%d %b %Y"),
        "n": random.randint(1, 5),
        "amount": random.randint(200, 1500),
        "balance": random.randint(3000, 40000),
        "agent": f"Agent {random.choice(['K. Walsh', 'M. Torres', 'P. Chen', 'D. Osei'])}",
        "adj": random.randint(15, 80),
    }
    if extra:
        ctx.update(extra)
    try:
        return template.format(**ctx)
    except KeyError:
        return template


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_borrower(rng: random.Random) -> dict:
    loan_type = rng.choice(LOAN_TYPES)
    fico = rng.randint(520, 820)
    dti = round(rng.uniform(0.20, 0.65), 2)
    gap = rng.choice([0, 0, 0, 1, 2, 3, 4, 6, 9, 12])
    late_30 = rng.randint(0, 5)
    late_60 = rng.randint(0, min(late_30, 3))
    late_90 = rng.randint(0, min(late_60, 2))
    deferrals = rng.randint(0, 4)
    ever_defaulted = rng.random() < 0.08

    b = {
        "borrower_id": f"B-{rng.randint(1000, 9999)}",
        "loan_id": f"L-{uuid.uuid4().hex[:8].upper()}",
        "loan_type": loan_type,
        "vintage_year": rng.randint(2019, 2024),
        "loan_amount": round(rng.uniform(5000, 450000), 2),
        "loan_term_months": rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360]),
        "fico_score": fico,
        "dti_ratio": dti,
        "ltv_ratio": round(rng.uniform(0.5, 1.1), 2) if loan_type == "mortgage" else None,
        "num_open_accounts": rng.randint(1, 12),
        "credit_history_yrs": round(rng.uniform(0.5, 25.0), 1),
        "payments_on_time": rng.randint(6, 48),
        "payments_late_30d": late_30,
        "payments_late_60d": late_60,
        "payments_late_90d": late_90,
        "num_deferrals": deferrals,
        "ever_defaulted": ever_defaulted,
        "employment_status": rng.choice(EMPLOYMENT_STATUSES),
        "employment_gap_months": gap,
        "annual_income": round(rng.uniform(28000, 280000), 2),
        "in_collections": late_90 >= 1 or ever_defaulted,
        "charged_off": ever_defaulted and rng.random() < 0.5,
        "origination_date": random_date(date(2019, 1, 1), date(2024, 1, 1)).isoformat(),
        "last_payment_date": random_date(date(2023, 1, 1), date(2024, 11, 1)).isoformat(),
        "record_updated_at": date.today().isoformat(),
    }
    b["analyst_risk_tier"] = assign_risk_tier(b)
    return b


def generate_notes_for_borrower(b: dict, rng: random.Random) -> list[dict]:
    notes = []
    num_notes = rng.randint(2, 5)
    note_type_pool = NOTE_TYPES if b["in_collections"] else ["underwriter", "servicing"]

    for i in range(num_notes):
        note_type = rng.choice(note_type_pool)
        template = rng.choice(NOTE_TEMPLATES[note_type])
        text = fill_template(template, b)

        notes.append({
            "note_id": f"N-{uuid.uuid4().hex[:10].upper()}",
            "borrower_id": b["borrower_id"],
            "loan_id": b["loan_id"],
            "note_type": note_type,
            "note_date": random_date(date(2020, 1, 1), date(2024, 11, 1)).isoformat(),
            "author_role": AUTHOR_ROLES[note_type],
            "note_text": text,
            "loan_type": b["loan_type"],
            "vintage_year": b["vintage_year"],
            "risk_band": b["analyst_risk_tier"],
            "created_at": date.today().isoformat(),
        })
    return notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n: int, seed: int):
    rng = random.Random(seed)
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    borrowers = [generate_borrower(rng) for _ in range(n)]
    all_notes = []
    for b in borrowers:
        all_notes.extend(generate_notes_for_borrower(b, rng))

    # Write borrowers
    borrower_path = Path("data/raw/borrowers.csv")
    with open(borrower_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=borrowers[0].keys())
        writer.writeheader()
        writer.writerows(borrowers)

    # Write notes
    notes_path = Path("data/raw/case_notes.csv")
    with open(notes_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_notes[0].keys())
        writer.writeheader()
        writer.writerows(all_notes)

    # Summary
    from collections import Counter
    tiers = Counter(b["analyst_risk_tier"] for b in borrowers)
    print(f"\n✅ Generated {n} borrowers → {borrower_path}")
    print(f"✅ Generated {len(all_notes)} case notes → {notes_path}")
    print(f"\nRisk tier distribution:")
    for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {tier:8s}: {tiers[tier]:3d}  ({tiers[tier]/n*100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of borrowers to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args.n, args.seed)
