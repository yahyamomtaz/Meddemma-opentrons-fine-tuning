# MedGemma Fine-Tuning for Opentrons Protocol Metadata

This is a LoRA adapter for `unsloth/medgemma-4b-it-bnb-4bit`. It was fine-tuned
using text-only English-language Opentrons OT-2 protocol descriptions to
produce machine-readable JSON containing the protocol title, categories,
labware, pipettes, modules, and reagents. It extracts protocol metadata; it
does not generate executable robot code.

<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made%20with%20unsloth.png" width="200" style="display: block; margin: 20px auto;" > 

This model is a fine-tuned version of [unsloth/medgemma-4b-it-bnb-4bit](https://huggingface.co/unsloth/medgemma-4b-it-bnb-4bit).
It has been trained using [TRL](https://github.com/huggingface/trl).


Model documentation: [MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma)

Base model: [unsloth/medgemma-4b-it-bnb-4bit](https://huggingface.co/unsloth/medgemma-4b-it-bnb-4bit)

Model on HF hub: [yayamomt/medgemma-4b-it-sft-lora-opentrons](https://huggingface.co/yayamomt/medgemma-4b-it-sft-lora-opentrons)

> **Research and assistive use only.** This adapter does not generate, validate, or approve executable robot protocols. Every output must be reviewed by a qualified laboratory professional and validated in the Opentrons Protocol Designer / simulation environment before use.

## Intended use

Use this model as a draft metadata-extraction assistant for natural-language protocol descriptions, for example inventory planning, protocol-library search, or human-in-the-loop annotation.

It is not intended for clinical decision-making, diagnostic use, autonomous liquid handling, safety-critical lab operations, or use where an incorrect labware, reagent, module, or pipette selection could cause harm.

## Task and output schema

Input: a natural-language Opentrons protocol description.

Output: one JSON object, without explanatory text or Markdown fences:

```json
{
  "title": "string",
  "categories": ["string"],
  "labware": ["string"],
  "pipettes": ["string"],
  "modules": ["string"],
  "reagents": ["string"]
}
```

The strings are model predictions, not a validated controlled vocabulary. They may be generic, duplicated, omitted, or unsupported by the input.

## Training data

The dataset was built from [Opentrons Protocol Library](https://github.com/Opentrons/Protocols).
Each supervised example contains a cleaned natural-language protocol
description as the instruction and its structured metadata JSON as the target.

* **Notes**: Each entry is an OpenTrons Protocol API v2 protocol from the
  lab-automation community. The upstream `README.md` description, labware
  list, pipettes, and categories are preserved, along with the file path so
  the original can be inspected.


The dataset builder combines three complementary protocol artifacts. Structured
runtime/API data take precedence over prose when available.

| Field | Primary source | Fallback |
| --- | --- | --- |
| Description | `protocols/protocols/<slug>/README.md`, `Description` section | `README.json` description |
| Title and categories | `README.json` | Protocol metadata title |
| Labware | Runtime/API artifact: `<slug>.ot2.apiv2.py.json` | `README.json`, then README `Labware` section |
| Pipettes | Runtime/API artifact `instruments` | `README.json`, then README `Pipettes` section |
| Modules | Inferred from runtime/API labware names | `README.json` module section |
| Reagents | `README.json` or README reagent sections | None |

The runtime artifact is preferred for labware and instruments because it
records what the protocol API loaded rather than relying only on prose mentions.

### Cleaning and split

The builder converts Markdown/HTML to plain text, removes embedded media and bare URLs,
retains link labels, removes headings, rules, and common notes/troubleshooting
boilerplate, and collapses excess blank lines. It discards descriptions shorter
than 30 characters and records containing neither labware nor pipette metadata.
Instrument IDs are converted to display names, and module names are inferred
from explicit keywords such as `magnetic module`, `temperature module`, and
`thermocycler`.

Valid examples were shuffled with seed 42 and split 80/20 by protocol. The
snapshot contains 820 records: 656 training and 164 evaluation records. No
`source_protocol` identifier occurs in both splits.

### Reagent enrichment

Because reagents have no runtime/API equivalent, a conservative catalog pass
matches reagent aliases from `data/catalog.csv` only when they occur explicitly
in the instruction. It normalizes Unicode, `µ`/`u`, links, URLs, casing, and
punctuation; restricts short aliases to their source protocol; deduplicates
generic and specific names; and records every decision in audit CSV files.

- 70 catalog reagent candidates were accepted and 7 rejected.
- 89 reagent labels were added to 83 training examples.
- 22 reagent labels were added to 21 evaluation examples.
- 400/656 training outputs and 90/164 evaluation outputs still have empty `reagents` lists after enrichment.
- The enriched copies and audit artifacts are under `data/train.jsonl`.

## Training configuration

| Setting | Value |
| --- | --- |
| Base model | `unsloth/medgemma-4b-it-bnb-4bit` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Epochs | 3 |
| Optimizer | Paged AdamW 8-bit |
| Learning rate | 1e-4 |
| Batch size / gradient accumulation | 2 / 4 |
| Maximum sequence length | 2048 |
| Hardware | NVIDIA H100 |

The final checkpoint at step 246 was also the selected best checkpoint, with
`eval_loss=0.35`. The completed run reported `train_loss=0.40`. These loss
values are optimization diagnostics, not task-quality or safety metrics.

## Evaluation

### Labelled evaluation split

The model was evaluated on all 164 source-disjoint records in `data/eval.jsonl`.

| Field | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Categories | 0.79 | 0.78 | 0.79 |
| Labware | 0.35 | 0.32 | 0.34 |
| Pipettes | 0.65 | 0.70 | 0.67 |
| Modules | 0.67 | 0.80 | 0.73 |
| Reagents | 0.42 | 0.18 | 0.25 |

### structured-output

```bash
{
  "title": "GNA Octea Prep",
  "categories": [
    "Sample Prep"
  ],
  "labware": [
    "Opentrons 96 Tip Rack 300 µL",
    "Opentrons 96 Tip Rack 1000 µL",
    "Opentrons 24 Tube Rack with Eppendorf 1.5 mL Safe-Lock Snapcap",
    "Opentrons 96 Filter Tip Rack 20 µL",
    "Opentrons 96 Well Aluminum Block with Generic PCR Strip 200 µL",
    "Opentrons 24 Well Aluminum Block with NEST 1.5 mL Snapcap",
    "Opentrons 6 Tube Rack with Falcon 50 mL Conical"
  ],
  "pipettes": [
    "P300 8-Channel Pipette (GEN2)",
    "P300 Single Channel Pipette (GEN2)"
  ],
  "modules": [
    "Magnetic Module",
    "Magnetic Module GEN2",
    "Temperature Module",
    "Temperature Module GEN2"
  ],
  "reagents": []
}
```

## License and attribution

This adapter is a derivative of MedGemma. Its use and redistribution are governed by the applicable [Health AI Developer Foundations terms](https://huggingface.co/google/medgemma-4b-it). Do not replace this notice with a permissive license unless you have confirmed that doing so is compatible with the base-model terms and the data provenance.

The underlying data derive from Opentrons protocol-library and catalog
artifacts. Verify the exact upstream revision and redistribution rights before
publishing the dataset or its enriched fields.

## Citation

The model has been trained using TRL.

```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```
---
base_model: unsloth/medgemma-4b-it-bnb-4bit
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
license: other
license_name: Health AI Developer Foundations Terms of Use
license_link: https://huggingface.co/google/medgemma-4b-it
tags:
  - lora
  - peft
  - unsloth
  - opentrons
  - laboratory-automation
  - structured-output
  - not-for-all-audiences
---
