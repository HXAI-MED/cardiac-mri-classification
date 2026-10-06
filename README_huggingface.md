---
pretty_name: Processed ACDC and M&Ms2 Cardiac MRI
license: other
task_categories:
  - image-classification
tags:
  - medical-imaging
  - cardiac-mri
  - magnetic-resonance-imaging
  - nifti
  - patient-level-classification
  - image-classification
---

# Processed ACDC and M&Ms2 Cardiac MRI

This dataset contains processed cardiac cine MRI data from the **Automated Cardiac
Diagnosis Challenge (ACDC)** and the **Multi-Centre, Multi-Vendor & Multi-Disease
Cardiac Image Segmentation (M&Ms2)** dataset. It is prepared for patient-level
cardiac pathology classification with the CineMA preprocessing conventions.

The files are intended for research and education. This repository is a processed
derivative of the source datasets; users must comply with the terms, access
conditions, and citation requirements of the original ACDC and M&Ms2 datasets.

## Dataset Details

### Dataset Description

- **Curated by:** Muhammad Turab
- **Shared by:** [turab45](https://huggingface.co/turab45)
- **Primary modalities:** Cardiac cine MRI, including short-axis (SAX) views
- **File format:** NIfTI images (`.nii.gz`) and CSV metadata
- **Intended task:** Multi-class cardiac pathology classification
- **License:** See [License and attribution](#license-and-attribution)

### Dataset Sources

- **ACDC:** [ACDC challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- **M&Ms:** [M&Ms challenge](https://www.ub.edu/mnms/)
- **Training code:** [Cardiac MRI classification repository](https://github.com/turab45/cnn-model-training)

## Uses

### Direct Use

The processed files can be used for:

- Training and evaluating supervised cardiac MRI classifiers.
- Reproducing the ACDC and M&Ms2 experiments in the accompanying training code.
- Comparing 2-D central-SAX and 3-D SAX model pipelines.
- Developing data-loading, validation, and patient-level evaluation workflows.

The dataset is not intended to replace clinical assessment or to support diagnosis
or treatment decisions.

### Out-of-Scope Use

Do not use this dataset for:

- Clinical diagnosis, triage, treatment planning, or other medical decisions.
- Evaluating a model as clinically safe or ready for deployment without independent
  validation, regulatory review, and domain-expert oversight.
- Re-identification, linkage with external personal records, or attempts to infer
  information about individual patients or data contributors.
- Claims of performance on populations, scanners, hospitals, or acquisition
  protocols that are not represented in the source datasets.

## Dataset Structure

The repository contains two processed subsets:

```text
acdc/
  train_metadata.csv
  test_metadata.csv
  train/
  test/
mnms2/
  train_metadata.csv
  val_metadata.csv
  test_metadata.csv
  train/
  val/
  test/
```

The exact directory names and additional metadata files should be treated as part
of the uploaded repository layout. Each patient directory contains processed cine
MRI volumes. The classification loaders use SAX ED and ES images. Metadata includes
patient identifiers, pathology labels, and the number of available slices where
required by the preprocessing pipeline.

### Labels and Splits

The expected patient-level labels and split sizes are:

| Subset | Classes | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| ACDC | DCM, HCM, MINF, NOR, RV | 90 | 10 | 50 |
| M&Ms2 | ARR, CIA, FALL, HCM, LV, NOR | 160 | 30 | 110 |

The split counts refer to patients, not individual images or slices. The ACDC
validation split is derived from the development/training metadata by the
preprocessing and experiment pipeline. Images from the same patient must not be
distributed across different splits.

For the 2-D central-SAX workflow, one deterministic central SAX slice is selected
at end-diastole (ED) and one at end-systole (ES). These two phase images are stacked
as a two-channel input. The 3-D workflow retains the corresponding volume data.

## Dataset Creation

### Curation Rationale

The dataset was processed to provide a reproducible input layout for cardiac MRI
classification experiments. Processing keeps the patient-level organization and
metadata needed for deterministic split validation and supports both 2-D and 3-D
experiments.

### Source Data

The source data are cardiac cine MRI examinations from the ACDC and M&Ms datasets.
The source datasets contain data collected from clinical and research cohorts and
include pathology categories defined by their respective challenge protocols.

#### Data Collection and Processing

The source datasets were processed using the CineMA data-preprocessing pipeline.
The resulting files use NIfTI image volumes and CSV metadata. The downstream
classification workflow performs intensity scaling and, during training, may apply
contrast adjustment, Gaussian noise, affine augmentation, spatial cropping, and
padding. These training-time augmentations are not part of the stored source files.

The processed dataset should be validated before use. In particular, check that
metadata identifiers match patient directories, that ED and ES files are present,
and that no patient occurs in more than one split.

#### Who are the source data producers?

The source images and labels were produced by the clinical and research groups
described by the ACDC and M&Ms challenge organizers. This processed repository does
not claim ownership of the original clinical data.

### Annotations

The pathology labels are inherited from the source datasets and are used as
classification targets. This repository does not add new clinical annotations.

#### Annotation process

See the original ACDC and M&Ms documentation for the source annotation protocols,
label definitions, and quality-control procedures.

#### Who are the annotators?

The original source-dataset organizers and clinical experts are responsible for
the source annotations. The identities and roles of individual annotators are not
specified in this processed-data card.

#### Personal and Sensitive Information

Cardiac MRI is medical data and must be treated as sensitive. Patient identifiers
in processed metadata are dataset identifiers, not permission to identify or link
patients. Do not attempt re-identification or external linkage. Users should
review the original dataset documentation and applicable institutional, ethical,
and legal requirements before downloading or redistributing the files.

## Bias, Risks, and Limitations

- The dataset is limited to the populations, institutions, scanners, vendors, and
  acquisition protocols represented by ACDC and M&Ms.
- Class frequencies and cohort composition may not represent clinical prevalence.
- Performance can change substantially across hospitals, vendors, field strengths,
  protocols, demographics, and image-quality conditions.
- A processed dataset can inherit errors, missing data, label noise, and selection
  bias from its source datasets.
- Patient-level splits do not guarantee independence from every possible external
  dataset or acquisition site.
- The labels are suitable for benchmarking the stated research tasks, not for
  establishing clinical severity or treatment recommendations.

### Recommendations

Report the source subset, patient-level split, preprocessing version, model,
augmentation settings, random seeds, and evaluation metrics. Use external and
site-aware validation where possible, inspect errors by class and acquisition
source, and involve qualified clinical experts before drawing clinical conclusions.

## License and Attribution

This repository is a processed derivative of ACDC and M&Ms2. The appropriate use
and redistribution terms are inherited from the original source datasets and may
not be replaced by a blanket license for this processed copy. Before publishing
models, derivatives, or analyses, review and follow the current terms supplied by
the ACDC and M&Ms organizers.

Please cite the original ACDC and M&Ms publications or challenge pages, and cite
the processing or training repository when using this processed release.

## Dataset Card Authors

Turab Ahmed (`turab45`)

## Contact

Open an issue or discussion in the Hugging Face dataset repository:
[turab45/acdc_mnms2_processed](https://huggingface.co/datasets/turab45/acdc_mnms2_processed).