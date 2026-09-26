# Classification training

The layout follows CineMA's classification package: dataset-specific folders
contain `dataset.py`, `model.py`, `train.py`, `eval.py`, and `config.yaml`.
Shared implementations live directly under `classification/`.

```text
training_code/
├── classification/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   ├── utils.py
│   ├── protocol.py          # Reads the fixed protocol definitions
│   ├── protocol.yaml        # Labels, splits, supported models, reference metrics
│   ├── acdc_2d/
│   │   ├── config.yaml      # Editable experiment settings
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── eval.py
│   ├── acdc_3d/
│   ├── acdc_distillation/
│   ├── mnms2_sax_2d/
│   ├── mnms2_3d/
│   ├── architecture_bank/
│   └── reproduction/
└── runtime.py
```

Experiments reuse shared files where appropriate. Distillation reuses ACDC 3-D
loading; the architecture bank and reproduction share their dataset pipeline.
Regression and segmentation remain in the original `cinema/` package.

## Configuration and running

Use the existing CineMA environment and run from the repository root.
Hydra loads the `config.yaml` next to the selected `train.py`, as in CineMA.
Edit that YAML or pass `key=value` overrides:

```bash
python -m training_code.classification.acdc_2d.train \
  logging.dir=/path/to/results \
  data.acdc_processed=/path/to/processed/acdc \
  'run.stages=[validate,train]' \
  'model.models_2d=[resnet18]' \
  'model.initializations_2d=[randinit]' \
  'run.seeds=[0]' \
  run.smoke=true
```

For full training, set `run.smoke=false`. Training overrides include
`train.n_epochs`, `train.lr`, `train.batch_size`, `train.batch_size_per_device`,
and `train.n_workers`. A `null` value preserves the experiment's original
CineMA training defaults, loaded from `cinema/classification/<dataset>/config.yaml`.
`logging.dir` is required. Hydra saves the resolved configuration under that
output directory and keeps the working directory unchanged.

| Experiment | Processed-data setting | Purpose |
| --- | --- | --- |
| `acdc_2d` | `data.acdc_processed` | Central SAX ED/ES slices |
| `acdc_3d` | `data.processed_dir` | Full SAX ED/ES volumes |
| `acdc_distillation` | `data.processed_dir` | ACDC 3-D students; also set `model.teacher_dir` |
| `mnms2_sax_2d` | `data.mnms2_processed` | Central SAX ED/ES slices |
| `mnms2_3d` | `data.processed_dir` | Full SAX ED/ES volumes |
| `architecture_bank` | `data.acdc_processed`, `data.mnms2_processed` | Multi-task architecture comparisons |
| `reproduction` | `data.acdc_processed`, `data.mnms2_processed` | Official reproduction and LAX-4C CNN benchmark |

Inspect the configuration without starting a run:

```bash
python -m training_code.classification.acdc_2d.train --cfg job
```

The old argparse flags (`--stages`, `--output-dir`, etc.) are replaced by YAML
settings. Existing cohort definitions, checkpoints, and result paths are preserved.
Fixed research definitions are in `classification/protocol.yaml`; they are
separate from per-run hyperparameters. Both this package and `cinema/` are needed.

## Analysis imports

```python
from training_code.classification.acdc_2d.dataset import ACDCMidSAX2DDataset
from training_code.classification.acdc_2d.model import ACDC2DClassifier
```

The SCAG and EDA notebook imports use this layout. Existing notebook outputs,
patient data, and checkpoints remain in place.
