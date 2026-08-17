"""
Opentrons Protocol Dataset Builder
===================================
Extracts clean, high-quality (instruction, output) pairs from the Opentrons
protocol library for fine-tuning a language model.

Data sources (in priority order):
  1. <slug>.ot2.apiv2.py.json  → ground-truth labware & instruments from the
                                  Python API runtime (most reliable)
  2. README.json               → title, categories, reagents
  3. protocols/<slug>/README.md→ clean description text (HTML/noise stripped)

Output schema (JSON per sample):
  {
    "instruction": "<system_prefix>\n<clean description>\n---",
    "output": {
      "title": "...",
      "categories": [...],
      "labware": [...],
      "pipettes": [...],
      "modules": [...],
      "reagents": [...]
    }
  }
"""

import os
import re
import json
import random
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROTOCOLS_BUILD_DIR = "protocols/protoBuilds"
PROTOCOLS_RAW_DIR   = "protocols/protocols"
DATA_DIR   = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
EVAL_FILE  = os.path.join(DATA_DIR, "eval.jsonl")
FULL_FILE  = os.path.join(DATA_DIR, "opentrons_dataset.jsonl")

EVAL_RATIO  = 0.2
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Prompt prefixes  (deterministic per slug via hash)
# ---------------------------------------------------------------------------
PROMPT_PREFIXES = [
    "Given the description below, produce the protocol metadata "
    "(labware, pipettes, modules, reagents, categories) as JSON.",

    "Extract the required labware, pipettes, modules, and reagents from the "
    "protocol description below and return a JSON object.",

    "Convert this Opentrons OT-2 experiment description into structured "
    "protocol metadata as JSON.",

    "Please produce a structured JSON definition for the following OT-2 "
    "protocol, listing all labware, pipettes, modules, and reagents.",

    "Analyse this protocol description and output the protocol metadata "
    "(title, categories, labware, pipettes, modules, reagents) as JSON.",
]

# ---------------------------------------------------------------------------
# HTML -> plain text stripper
# ---------------------------------------------------------------------------
class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self._parts = []

    def handle_data(self, d):
        self._parts.append(d)

    def get_text(self):
        return "".join(self._parts)


def strip_html(text):
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


# ---------------------------------------------------------------------------
# Markdown / text cleaning helpers
# ---------------------------------------------------------------------------
def clean_description(text):
    """
    Turn raw README description markdown into clean plain text suitable for
    an instruction prompt.
    """
    if not text:
        return ""

    # Strip HTML
    text = strip_html(text)

    # Remove markdown images completely
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Collapse markdown hyperlinks -> keep label text only (handles empty URLs too)
    text = re.sub(r"\[([^\]]+)\]\s*\([^)]*\)", r"\1", text)

    # Remove bare URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove Opentrons boilerplate "Notes" / troubleshooting blurb
    text = re.sub(
        r"(?:Notes?|If you have any questions).*?(?=\n\n|\Z)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove bullet lines that became empty after link stripping
    text = re.sub(r"^\s*[\*\-]\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ blank lines -> 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_section(markdown, section_name):
    """Extract the content of a named ## section from markdown text."""
    pattern = r"#+\s*" + re.escape(section_name) + r"[^\n]*\n(.*?)(?=\n#+|\Z)"
    m = re.search(pattern, markdown, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_bullet_list(text):
    """Parse a markdown bullet list into a clean list of strings."""
    items = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[\*\-\+]\s*", "", line).strip()
        if not line:
            continue
        # Remove markdown images within bullet lines
        line = re.sub(r"!\[.*?\]\(.*?\)", "", line).strip()
        # Collapse links to label text (handles empty URLs and space-separated variants)
        line = re.sub(r"\[([^\]]+)\]\s*\([^)]*\)", r"\1", line)
        # Remove bare URLs
        line = re.sub(r"https?://\S+", "", line).strip()
        # Skip noise lines
        skip_patterns = [
            r"opentrons\.com",
            r"troubleshooting",
            r"protocol-troubleshooting",
            r"paperform",
        ]
        if any(re.search(p, line, re.IGNORECASE) for p in skip_patterns):
            continue
        # Skip lines that start with ! (leftover image refs like "!reagent calculator")
        if line.startswith("!"):
            continue
        # Skip lines that are just punctuation or numbers
        if re.match(r"^[\d\W]+$", line):
            continue
        # Skip if line is still just a bracketed fragment (parsing artifact)
        if re.match(r"^\[.*\]$", line):
            continue
        if len(line) > 3:
            items.append(line)
    return items


# ---------------------------------------------------------------------------
# Instrument name normaliser
# ---------------------------------------------------------------------------
INSTRUMENT_DISPLAY = {
    "p10_single":          "P10 Single Channel Pipette",
    "p10_multi":           "P10 8-Channel Pipette",
    "p20_single_gen2":     "P20 Single Channel Pipette (GEN2)",
    "p20_multi_gen2":      "P20 8-Channel Pipette (GEN2)",
    "p50_single":          "P50 Single Channel Pipette",
    "p50_multi":           "P50 8-Channel Pipette",
    "p300_single":         "P300 Single Channel Pipette",
    "p300_multi":          "P300 8-Channel Pipette",
    "p300_single_gen2":    "P300 Single Channel Pipette (GEN2)",
    "p300_multi_gen2":     "P300 8-Channel Pipette (GEN2)",
    "p1000_single":        "P1000 Single Channel Pipette",
    "p1000_single_gen2":   "P1000 Single Channel Pipette (GEN2)",
    "p1000_multi_flex":    "P1000 8-Channel Pipette (Flex)",
    "p50_single_flex":     "P50 Single Channel Pipette (Flex)",
    "p50_multi_flex":      "P50 8-Channel Pipette (Flex)",
    "p200_single":         "P200 Single Channel Pipette",
    "p200_multi":          "P200 8-Channel Pipette",
}

MODULE_KEYWORDS = {
    "magnetic module gen2":       "Magnetic Module GEN2",
    "magnetic module":            "Magnetic Module",
    "magdeck":                    "Magnetic Module",
    "temperature module gen2":    "Temperature Module GEN2",
    "temperature module":         "Temperature Module",
    "tempdeck":                   "Temperature Module",
    "thermocycler module":        "Thermocycler Module",
    "thermocycler":               "Thermocycler Module",
    "heater-shaker module":       "Heater-Shaker Module",
    "heater-shaker":              "Heater-Shaker Module",
    "absorbance plate reader":    "Absorbance Plate Reader",
    "flex stacker":               "Flex Stacker",
}


def normalise_instrument(name):
    key = name.lower().strip()
    return INSTRUMENT_DISPLAY.get(key, name.title())


def extract_modules_from_labware(labware_list):
    """Infer modules from labware slot descriptions in the py.json."""
    modules = set()
    for lw in labware_list:
        name_lower = lw.get("name", "").lower()
        for key, display in MODULE_KEYWORDS.items():
            if key in name_lower:
                modules.add(display)
    return sorted(modules)


def extract_labware_names(labware_list):
    """Convert py.json labware list to clean display names, excluding trash."""
    trash_patterns = [r"trash"]
    seen = set()
    names = []
    for lw in labware_list:
        name = lw.get("name", "")
        # Strip slot suffix: "NEST 96 Deepwell Plate on 1" -> "NEST 96 Deepwell Plate"
        clean = re.sub(r"\s+on\s+.*$", "", name, flags=re.IGNORECASE).strip()
        if any(re.search(p, clean, re.IGNORECASE) for p in trash_patterns):
            continue
        if clean and clean not in seen:
            seen.add(clean)
            names.append(clean)
    return names


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def process_protocol(slug):
    """
    Extract a single (instruction, output) sample from a protocol slug.
    Returns None if the sample should be skipped.
    """
    build_path = os.path.join(PROTOCOLS_BUILD_DIR, slug)
    if not os.path.isdir(build_path):
        return None

    readme_json_path = os.path.join(build_path, "README.json")
    if not os.path.exists(readme_json_path):
        return None

    # Load README.json
    try:
        with open(readme_json_path, encoding="utf-8") as f:
            readme = json.load(f)
    except Exception:
        return None

    # Load the .py.json (ground-truth API output)
    py_json_files = [
        fn for fn in os.listdir(build_path)
        if fn.endswith(".json") and "README" not in fn and "metadata" not in fn
    ]
    py_data = {}
    if py_json_files:
        try:
            with open(os.path.join(build_path, py_json_files[0]), encoding="utf-8") as f:
                py_data = json.load(f)
        except Exception:
            pass

    # Load raw README.md for clean description
    raw_md_path = os.path.join(PROTOCOLS_RAW_DIR, slug, "README.md")
    raw_md = ""
    if os.path.exists(raw_md_path):
        try:
            with open(raw_md_path, encoding="utf-8") as f:
                raw_md = f.read()
        except Exception:
            pass

    # -- Description ----------------------------------------------------------
    desc = ""
    if raw_md:
        desc = extract_section(raw_md, "Description")

    if not desc:
        desc = readme.get("markdown", {}).get("description") or readme.get("description", "")

    desc = clean_description(desc)
    if len(desc.strip()) < 30:
        return None  # Too short to be informative

    # -- Title ----------------------------------------------------------------
    title = (
        readme.get("title")
        or (py_data.get("metadata") or {}).get("protocolName")
        or ""
    ).strip()

    # -- Categories -----------------------------------------------------------
    cats = readme.get("categories", {})
    if isinstance(cats, dict):
        cat_list = list(cats.keys())
    elif isinstance(cats, list):
        cat_list = [str(c) for c in cats]
    else:
        cat_list = []

    # -- Labware (py.json is ground truth) ------------------------------------
    labware_names = []
    if py_data.get("labware"):
        labware_names = extract_labware_names(py_data["labware"])

    if not labware_names:
        lw_md = readme.get("markdown", {}).get("labware") or readme.get("labware", "")
        labware_names = parse_bullet_list(lw_md) if lw_md else []

    if not labware_names and raw_md:
        lw_raw = extract_section(raw_md, "Labware")
        labware_names = parse_bullet_list(lw_raw)

    # -- Pipettes (py.json instruments are ground truth) ----------------------
    pipette_names = []
    if py_data.get("instruments"):
        pipette_names = [normalise_instrument(i["name"]) for i in py_data["instruments"]]

    if not pipette_names:
        pip_md = readme.get("markdown", {}).get("pipettes") or readme.get("pipettes", "")
        pipette_names = parse_bullet_list(pip_md) if pip_md else []

    if not pipette_names and raw_md:
        pip_raw = extract_section(raw_md, "Pipettes")
        pipette_names = parse_bullet_list(pip_raw)

    # -- Modules (inferred from py.json labware slot names) -------------------
    modules = []
    if py_data.get("labware"):
        modules = extract_modules_from_labware(py_data["labware"])

    if not modules:
        mod_md = readme.get("markdown", {}).get("modules") or readme.get("modules", "")
        modules = parse_bullet_list(mod_md) if mod_md else []

    # -- Reagents (only in README markdown - no API equivalent) ---------------
    reagent_names = []
    reagents_raw = readme.get("reagents")
    if reagents_raw:
        if isinstance(reagents_raw, list):
            # Each item may contain markdown links — clean them
            combined = "\n".join(f"* {r}" for r in reagents_raw)
            reagent_names = parse_bullet_list(combined)
        elif isinstance(reagents_raw, str) and reagents_raw.strip():
            reagent_names = parse_bullet_list(reagents_raw)

    if not reagent_names:
        reagent_md = readme.get("markdown", {}).get("reagents") or ""
        reagent_names = parse_bullet_list(reagent_md)

    if not reagent_names and raw_md:
        for section in ["Reagents", "Reagent Setup", "Reagent-Setup"]:
            raw_section = extract_section(raw_md, section)
            if raw_section:
                reagent_names = parse_bullet_list(raw_section)
                if reagent_names:
                    break

    # -- Quality filter -------------------------------------------------------
    if not labware_names and not pipette_names:
        return None

    # -- Build instruction ----------------------------------------------------
    prefix = PROMPT_PREFIXES[hash(slug) % len(PROMPT_PREFIXES)]
    instruction = f"{prefix}\n{desc}\n---"

    # -- Build output JSON ----------------------------------------------------
    output_dict = {
        "title":      title,
        "categories": cat_list,
        "labware":    labware_names,
        "pipettes":   pipette_names,
        "modules":    modules,
        "reagents":   reagent_names,
    }

    return {
        "instruction":     instruction,
        "output":          json.dumps(output_dict, indent=2, ensure_ascii=False),
        "source_protocol": slug,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    slugs = sorted(os.listdir(PROTOCOLS_BUILD_DIR))
    entries = []
    skipped = 0

    for slug in slugs:
        result = process_protocol(slug)
        if result is None:
            skipped += 1
        else:
            entries.append(result)

    print(f"Processed {len(slugs)} protocols")
    print(f"  Valid samples : {len(entries)}")
    print(f"  Skipped       : {skipped}  (missing labware+pipettes or too-short description)")

    # Reproducible shuffle & split
    random.seed(RANDOM_SEED)
    random.shuffle(entries)

    num_eval   = int(len(entries) * EVAL_RATIO)
    eval_entries  = entries[:num_eval]
    train_entries = entries[num_eval:]

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for e in train_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for e in eval_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(FULL_FILE, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n  Train -> {TRAIN_FILE}  ({len(train_entries)} samples)")
    print(f"  Eval  -> {EVAL_FILE}   ({len(eval_entries)} samples)")
    print(f"  Full  -> {FULL_FILE}   ({len(entries)} samples)")

    # Quick quality report
    import statistics
    desc_lens   = [len(e["instruction"]) for e in entries]
    output_lens = [len(e["output"])      for e in entries]

    has_labware  = sum(1 for e in entries if json.loads(e["output"])["labware"])
    has_pipettes = sum(1 for e in entries if json.loads(e["output"])["pipettes"])
    has_modules  = sum(1 for e in entries if json.loads(e["output"])["modules"])
    has_reagents = sum(1 for e in entries if json.loads(e["output"])["reagents"])

    print("\n-- Quality report -------------------------------------------------")
    print(f"  Instruction length  avg={statistics.mean(desc_lens):.0f}  "
          f"min={min(desc_lens)}  max={max(desc_lens)}")
    print(f"  Output length       avg={statistics.mean(output_lens):.0f}  "
          f"min={min(output_lens)}  max={max(output_lens)}")
    print(f"  Has labware   : {has_labware}/{len(entries)}  ({100*has_labware/len(entries):.1f}%)")
    print(f"  Has pipettes  : {has_pipettes}/{len(entries)}  ({100*has_pipettes/len(entries):.1f}%)")
    print(f"  Has modules   : {has_modules}/{len(entries)}  ({100*has_modules/len(entries):.1f}%)")
    print(f"  Has reagents  : {has_reagents}/{len(entries)}  ({100*has_reagents/len(entries):.1f}%)")
