# Cardiac MRI classification training

Train CNN variants using CineMA's ACDC and M&Ms2 preprocessing, patient splits,
augmentation settings, optimizer parameters, and learning-rate schedule.
Select an experiment in `configs/`; the CNN architecture changes while the task
setup comes directly from the installed CineMA checkout.

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
│   ├── models_2d.py
│   ├── models_3d.py
│   ├── models.py          # Existing notebook/model imports
│   ├── train.py
│   ├── experiments.py
│   ├── evaluate.py
│   ├── results.py
│   ├── distillation.py
│   ├── cinema_support.py
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

Use Python 3.11. Keep CineMA as an installed dependency: its source does not need
to be copied into your project. `requirements.txt` pins CineMA to the tested
commit `c10daa1d93f0ea28d8b9ad9206b0f673d25805c1` and installs it as an editable
checkout so its subpackages and task YAML files are available.

Create and activate an environment, then install the project requirements:

```bash
uv venv --python 3.11 --seed
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The tested environment uses PyTorch `2.5.1+cu121`, torchvision `0.20.1+cu121`,
and MONAI `1.5.2`. Install the appropriate PyTorch CUDA build for your machine.
Your existing `cinema` environment can also run this project.

`requirements.txt` is the installation source of truth; it is not a complete
environment lockfile. `pyproject.toml` records project metadata and Python support.

## Processed dataset

The processed ACDC and M&Ms2 datasets are publicly available on Hugging Face:
[`turab45/acdc_mnms2_processed`](https://huggingface.co/datasets/turab45/acdc_mnms2_processed).
Download the dataset and point the relevant config values to the extracted ACDC
or M&Ms2 directory, such as `data.acdc_processed` or `data.mnms2_processed`.

The configs use the relocated local data and model directories by default:

```text
/media/kislay/New Volume/Turab/0. Code workspace/Data and models/
├── datasets/                 # Previously processed/
│   ├── acdc/
│   └── mnms2/
└── model checkpoints/        # Previously architecture_bank/
    ├── 2d/                  # Existing model checkpoints
    └── 3d/
```

Change the processed-data settings and `logging.dir` when running on another
machine. Quote Hydra path overrides containing spaces, for example
`'data.acdc_processed="/path with spaces/datasets/acdc"'`.

## Run

For a step-by-step single-model example, open
[`notebooks/single_model_training.ipynb`](notebooks/single_model_training.ipynb).
It trains one ACDC ResNet18 with one seed, covers validation and result inspection,
and reloads the checkpoint for patient inference. It uses the relocated ACDC
dataset and a `single_model_notebook/` folder under `model checkpoints/`.
Use a Python 3.11 kernel with CineMA and the project dependencies installed.
Set `SMOKE = True` for a one-epoch check; `SMOKE = False` uses the notebook's
explicit training settings.

For Grad-CAM and segmentation overlays, open
[`notebooks/cmri-gradcma.ipynb`](notebooks/cmri-gradcma.ipynb). It restores the
relocated ACDC ResNet50 checkpoint, reuses evaluation preprocessing, and exports
PNGs to `outputs/gradcam/`. Its final sections also restore the M&Ms2 four-chamber
LAX checkpoint, run ED/ES inference and Grad-CAM, and save original images and
overlays under `outputs/gradcam/mnms2_lax_4c/`. Adjust `LAX_CLASS_NAME`,
`LAX_N_SAMPLES`, and `LAX_CHECKPOINT_PATH` in the first cell.

Run commands from this project's root; the checkout folder can have any name.
Inspect a config without loading models or patient data:

```bash
python main.py --config-name acdc_2d --cfg job
```

Validate and run a one-epoch smoke experiment:

```bash
python main.py --config-name acdc_2d \
  'run.stages=[validate,train]' \
  'model.models_2d=[resnet18]' \
  'model.initializations_2d=[randinit]' \
  'run.seeds=[0]' \
  run.smoke=true
```

For full training, use `run.smoke=false`. Change YAML settings directly or use
Hydra `key=value` overrides. Outputs default to
`/media/kislay/New Volume/Turab/0. Code workspace/Data and models/model checkpoints/`;
change the root with `logging.dir=/path/to/results`.
Hydra keeps the working directory unchanged.
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
For the existing processed datasets, use `'run.stages=[calibrate]'` with the
architecture-bank and reproduction configs; preprocessing requires raw-data paths.

A `null` training value inherits the installed CineMA task config. Fixed class
orders, split sizes, model options, and published reference scores live in
`protocol.yaml`. Patient split logic follows CineMA and checks for leakage.

The full-volume and LAX experiments reuse CineMA's dataset and transforms directly.
The central-SAX 2-D experiments adapt its SAX settings to two dimensions and select
one deterministic central ED/ES slice per patient. ImageNet initialization adds
the existing grayscale normalization. ACDC 3-D random-initialization and student
experiments retain the protocol's explicit `0.01` weight decay unless overridden.

## Outputs

New artifacts are written under `logging.dir`, with the experiment name before
their existing run hierarchy. With the default path:

```text
Data and models/model checkpoints/checkpoints/acdc_2d/architecture_bank/2d/acdc_sax_mid_2d/
    resnet18/stacked/randinit/seed_0/
```

- `checkpoints/`: weights, architecture descriptions, resume summaries and failure records.
- `metrics/`: evaluation scores, confusion matrices, and aggregate tables.
- `predictions/`: patient-level probability CSVs for validation, test, and ensembles.
- `logs/`: training histories, launch commands, and Hydra configs/logs.

Preprocessing, split audits, Hugging Face downloads, and distillation targets may
also create `processed/`, `splits/`, `hf_cache/`, and `teacher_targets/` under the
output root. Official CineMA subprocesses keep their own native run files inside
the checkpoint run directory; their aggregate reports go to `metrics/`.
Generated outputs and Python caches are ignored by Git.

Each run saves `run_settings.json` with the effective CineMA config, source commit,
split-metadata fingerprints, and model/training options. `run.resume=true` reuses
a completed run only when these settings match and its checkpoint exists.
Changing settings in an occupied directory raises an error; use a new
`logging.dir` to retain both runs. `run.resume=false` retrains a matching run from
the beginning; it does not continue an interrupted optimizer state.

Previously trained models are directly under `model checkpoints/2d/` and
`model checkpoints/3d/`. New runs use the `checkpoints/<experiment>/` hierarchy
shown above. The experiment namespace prevents
ACDC/M&Ms2-specific models from colliding with the architecture-bank implementations.
Use the new paths for training; do not move old summaries into them to bypass checks.

## Code and migration

Follow a run in this order:

1. `main.py` loads the selected YAML configuration.
2. `src/experiments.py` chooses the experiment and requested stages.
3. `src/cinema_support.py` reads CineMA's task defaults and invokes its original
   preprocessing or official-training modules.
4. `src/dataset.py` builds patient datasets and validates the splits.
5. `src/models_2d.py` or `src/models_3d.py` builds the requested CNN.
6. `src/train.py` runs the shared CNN optimization, validation, early stopping,
   and checkpoint selection. It includes the final partial accumulation group.
7. `src/evaluate.py` exports predictions; `src/results.py` combines completed seeds.

`src/distillation.py` contains teacher targets, student training, and comparisons.
`src/metrics.py` computes scores; `src/utils.py` handles settings, seeds and paths.
`src/models.py` preserves the model imports used by existing notebooks.

To add a CNN, extend the appropriate model builder and its allowed choices in
`protocol.yaml`, then select it in an experiment YAML. Reuse the existing dataset
and trainer. Keep changes to CineMA itself in a separate checkout or fork.

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

The tests cover configs, metrics, output routing, import consistency, gradient
accumulation, checkpoint selection, safe result reuse, and agreement with CineMA's
task defaults and patient splits. They use synthetic data and CPU training;
CineMA and the project dependencies must be installed. A complete research-data
smoke run additionally requires the processed datasets.
