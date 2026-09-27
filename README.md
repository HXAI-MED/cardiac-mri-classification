# Cardiac MRI classification training

A configuration-driven entry point for ACDC and M&Ms2 CNN experiments.
Select an experiment in `configs/`; shared code is organized by responsibility
in `src/`. Dataset-specific functions have explicit prefixes where protocols differ.

```text
training_code/
├── configs/
│   ├── acdc_2d.yaml
│   ├── acdc_3d.yaml
│   ├── acdc_distillation.yaml
│   ├── mnms2_sax_2d.yaml
│   ├── mnms2_3d.yaml
│   ├── architecture_bank.yaml
│   └── reproduction.yaml
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── protocol.py
│   └── utils.py
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   ├── predictions/
│   └── logs/
├── tests/
├── protocol.yaml
├── main.py
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or 3.11 and the CineMA environment used for your experiments.
This repository requires the separate `cinema` package for preprocessing,
training defaults, transforms, optimizers, and ConvViT/ResNet implementations.
Install that source checkout with its own dependencies first, then:

```bash
python -m pip install -r requirements.txt
```

The requirements list direct dependencies; it is not a reproducibility lockfile.
Keep the original environment's compatible PyTorch/torchvision versions for
research runs. Python 3.14 is not supported by the Hydra launcher used here.

## Run

Run commands from this project's root; the checkout folder can have any name.
Inspect a config without loading models or patient data:

```bash
python main.py --config-name acdc_2d --cfg job
```

Validate and run a one-epoch smoke experiment:

```bash
python main.py --config-name acdc_2d \
  data.acdc_processed=/path/to/processed/acdc \
  'run.stages=[validate,train]' \
  'model.models_2d=[resnet18]' \
  'model.initializations_2d=[randinit]' \
  'run.seeds=[0]' \
  run.smoke=true
```

For full training, use `run.smoke=false`. Change YAML settings directly or use
Hydra `key=value` overrides. Outputs default to `outputs/`; change the root with
`logging.dir=/path/to/results`. Hydra keeps the working directory unchanged.
Relative data/output paths are relative to the working directory.

| Config | Processed-data setting | Experiment |
| --- | --- | --- |
| `acdc_2d` | `data.acdc_processed` | Central SAX ED/ES slices |
| `acdc_3d` | `data.processed_dir` | Full SAX ED/ES volumes |
| `acdc_distillation` | `data.processed_dir` | ACDC 3-D students; also set `model.teacher_dir` |
| `mnms2_sax_2d` | `data.mnms2_processed` | Central SAX ED/ES slices |
| `mnms2_3d` | `data.processed_dir` | Full SAX ED/ES volumes |
| `architecture_bank` | `data.acdc_processed`, `data.mnms2_processed` | Multi-task model comparisons |
| `reproduction` | `data.acdc_processed`, `data.mnms2_processed` | Official reproduction and LAX-4C benchmark |

The five dataset/student configs default to validation only. The architecture
bank and reproduction retain their preprocessing/calibration defaults. Stage
choices are defined in `protocol.yaml`. Training includes validation and final
test evaluation; full-volume configs also support `run.stages=[aggregate]`.

A `null` training value inherits the installed CineMA task config. Fixed class
orders, patient splits, model options, and published reference scores live in
`protocol.yaml`. These remain separate from per-run settings.

## Outputs

Artifacts retain their task/model/formulation/initialization/seed hierarchy:

- `checkpoints/`: weights, architecture descriptions, resume summaries and failure records.
- `metrics/`: evaluation scores, confusion matrices, and aggregate tables.
- `predictions/`: patient-level probability CSVs for validation, test, and ensembles.
- `logs/`: training histories, launch commands, and Hydra configs/logs.

Preprocessing, split audits, Hugging Face downloads, and distillation targets may
also create `processed/`, `splits/`, `hf_cache/`, and `teacher_targets/` under the
output root. Official CineMA subprocesses keep their own native run files inside
the checkpoint run directory; their aggregate reports go to `metrics/`.
Generated outputs and Python caches are ignored by Git.

Resume uses summaries in the new checkpoint layout. Existing external result
folders are not moved or automatically migrated by this restructuring.

## Code and migration

`src/dataset.py` handles data and leakage checks; `src/models.py` defines models;
`src/train.py` dispatches experiments and runs optimization; `src/evaluate.py`
handles inference/export/aggregation; `src/metrics.py` computes scores;
`src/utils.py` handles configuration, reproducibility, and output paths.

The former `classification/<experiment>/` modules and root `runtime.py` have been
consolidated. Replace old module launch commands with `python main.py
--config-name <experiment>`. Update notebook imports, for example:

```python
from src.dataset import ACDCMidSAX2DDataset
from src.models import ACDC2DClassifier, ACDC3DClassifier, MnMs2SAX3DClassifier
from src.metrics import metrics_from_probabilities
```

Use the project root on the notebook's Python path. Model state-dict structure,
training schedules, class ordering, and cohort definitions are retained.

## Checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q src main.py
```

The tests cover configs/overrides, class-specific metrics, output routing, and
internal import consistency without requiring GPUs or patient data. A complete
training smoke run additionally requires CineMA and the processed datasets.
