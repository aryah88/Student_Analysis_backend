# backend/models/generate_synthetic.py
import csv
import random
import math
from pathlib import Path

OUT = Path(__file__).parent / "sample_train.csv"

def make_student(i):
    # features: comprehension, attention, focus, retention, engagement_time
    comprehension = random.uniform(30, 100)
    attention = random.uniform(20, 100)
    focus = random.uniform(20, 100)
    retention = random.uniform(10, 100)
    engagement_time = random.uniform(5, 120)  # minutes

    # create a target (assessment_score) realistic-ish
    # this is synthetic: weighted combination + noise
    score = (0.28*comprehension + 0.22*attention + 0.2*focus + 0.18*retention + 0.12*(engagement_time/1.2))
    score += random.gauss(0, 6)  # noise
    score = max(0, min(100, score))

    # persona synthetic (0,1,2) example based on profile:
    if comprehension > 80 and attention > 70:
        persona = 1
    elif engagement_time < 25:
        persona = 0
    else:
        persona = 2

    return {
        "student_id": i,
        "name": f"Student_{i}",
        "class": random.choice(["A", "B", "C"]),
        "comprehension": round(comprehension, 6),
        "attention": round(attention, 6),
        "focus": round(focus, 6),
        "retention": round(retention, 6),
        "engagement_time": round(engagement_time, 6),
        "assessment_score": round(score, 6),
        "persona": persona
    }

def generate(n=200, out=OUT):
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["student_id","name","class","comprehension","attention","focus","retention","engagement_time","assessment_score","persona"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, n+1):
            writer.writerow(make_student(i))
    print(f"Wrote {n} rows to {out}")

if __name__ == "__main__":
    generate(400)  # make 400 rows by default
