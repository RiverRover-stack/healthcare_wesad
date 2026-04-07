"""
Filters papers_WESAD.txt down to the 32 relevant papers.
Run: python filter_papers.py

Output: outputs/reports/papers_filtered.bib  (32 papers)
        outputs/reports/papers_removed.bib   (65 removed papers)
"""

import re
import os

INPUT_FILE  = "papers_WESAD.txt"
OUTPUT_KEEP = "outputs/reports/papers_filtered.bib"
OUTPUT_CUT  = "outputs/reports/papers_removed.bib"

# ── The 32 papers to KEEP ─────────────────────────────────────────────────────
KEEP_KEYS = {
    # Direct WESAD / Stress Detection
    "benita2024stress",
    "li2020stress",
    "dziezyc2020can",
    "abdelfattah2025machine",
    "xiang2025multi",
    "behinaein2021transformer",
    "kang2021classification",
    "rashid2023stress",
    "siirtola2020comparison",

    # Knowledge Distillation
    "liu2023emotionkd",
    "cao2024online",
    "wang2024adaptive",
    "xu2022contrastive",
    "wang2024lightweight",

    # Explainability / SHAP
    "moser2024explainable",
    "banerjee2023heart",
    "shikha2024optimization",
    "abdelaal2024exploring",
    "adarsh2024mental",
    "jaber2022medically",

    # TinyML / Edge Deployment
    "gibbs2023combining",
    "rostami2025real",
    "heydari2025tiny",
    "abadade2023comprehensive",
    "lamaakal2025comprehensive",

    # Multi-Scale CNN Architecture
    "wang2020deep",
    "yang2023multi",

    # Survey / Review
    "gedam2021review",
    "kyrou2024deep",
    "taskasaplidis2024review",

    # Cross-Subject Generalization
    "benchekroun2023cross",
    "rabbani2022contrastive",
}

# ── Parse .bib file into individual entries ───────────────────────────────────
def parse_bib(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on @ but keep the @ sign
    raw_entries = re.split(r'(?=@\w+\{)', content)
    entries = {}
    for entry in raw_entries:
        entry = entry.strip()
        if not entry:
            continue
        # Extract citation key
        match = re.match(r'@\w+\{(\S+),', entry)
        if match:
            key = match.group(1).rstrip(',')
            entries[key] = entry
    return entries

# ── Main ──────────────────────────────────────────────────────────────────────
os.makedirs("outputs/reports", exist_ok=True)

entries = parse_bib(INPUT_FILE)
print(f"Total papers found in file: {len(entries)}")

kept   = {k: v for k, v in entries.items() if k in KEEP_KEYS}
missed = KEEP_KEYS - set(entries.keys())
cut    = {k: v for k, v in entries.items() if k not in KEEP_KEYS}

# Write filtered .bib
with open(OUTPUT_KEEP, "w", encoding="utf-8") as f:
    f.write("% ============================================================\n")
    f.write("% Filtered bibliography — 32 core papers for WESAD KD paper\n")
    f.write("% ============================================================\n\n")

    f.write("% ── ADD THESE 3 MANUALLY (not in original library) ──────────\n")
    f.write("% Schmidt et al. 2018 — WESAD dataset paper (ACM ICMI)\n")
    f.write("% Hinton et al. 2015 — Distilling the Knowledge in a Neural Network\n")
    f.write("% Lundberg & Lee 2017 — A Unified Approach to Interpreting Model Predictions (SHAP)\n\n")

    categories = {
        "Direct WESAD / Stress Detection": [
            "benita2024stress","li2020stress","dziezyc2020can",
            "abdelfattah2025machine","xiang2025multi","behinaein2021transformer",
            "kang2021classification","rashid2023stress","siirtola2020comparison"
        ],
        "Knowledge Distillation": [
            "liu2023emotionkd","cao2024online","wang2024adaptive",
            "xu2022contrastive","wang2024lightweight"
        ],
        "Explainability / SHAP": [
            "moser2024explainable","banerjee2023heart","shikha2024optimization",
            "abdelaal2024exploring","adarsh2024mental","jaber2022medically"
        ],
        "TinyML / Edge Deployment": [
            "gibbs2023combining","rostami2025real","heydari2025tiny",
            "abadade2023comprehensive","lamaakal2025comprehensive"
        ],
        "Multi-Scale CNN Architecture": [
            "wang2020deep","yang2023multi"
        ],
        "Survey / Review Papers": [
            "gedam2021review","kyrou2024deep","taskasaplidis2024review"
        ],
        "Cross-Subject Generalization": [
            "benchekroun2023cross","rabbani2022contrastive"
        ],
    }

    for category, keys in categories.items():
        f.write(f"% ── {category} ──────────────────────────────────────────\n")
        for key in keys:
            if key in kept:
                f.write(kept[key] + "\n\n")
            else:
                f.write(f"% WARNING: {key} not found in input file\n\n")

with open(OUTPUT_CUT, "w", encoding="utf-8") as f:
    f.write("% Papers removed during triage — do not cite\n\n")
    for key, entry in cut.items():
        f.write(entry + "\n\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n✓ Kept:    {len(kept)} papers  → {OUTPUT_KEEP}")
print(f"  Cut:     {len(cut)} papers  → {OUTPUT_CUT}")
if missed:
    print(f"\n⚠ These keys were in KEEP list but NOT found in your file:")
    for k in sorted(missed):
        print(f"    {k}")
print(f"\n+ Add 3 mandatory papers manually:")
print(f"    Schmidt et al. 2018 (WESAD dataset)")
print(f"    Hinton et al. 2015 (Knowledge Distillation)")
print(f"    Lundberg & Lee 2017 (SHAP)")
