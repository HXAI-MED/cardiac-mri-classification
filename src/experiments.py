"""Select experiments, run requested stages and record failures."""

from __future__ import annotations

import argparse
import gc
import platform
import sys
import traceback

import torch
from omegaconf import DictConfig

from .cinema_support import preprocess_dataset, processed_dir, run_official_training
from .dataset import (
    acdc_2d_load_official_splits,
    acdc_2d_resolve_processed_dir,
    acdc_2d_save_split_audit,
    acdc_2d_validate_inputs,
    acdc_3d_preprocess,
    acdc_3d_processed_dir,
    acdc_3d_validate_inputs,
    load_acdc_config,
    load_mnms2_config,
    mnms2_3d_preprocess,
    mnms2_3d_processed_dir,
    mnms2_3d_validate_inputs,
    mnms2_sax_2d_load_official_splits,
    mnms2_sax_2d_resolve_processed_dir,
    mnms2_sax_2d_save_split_audit,
    mnms2_sax_2d_validate_inputs,
    preprocess_acdc,
    preprocess_mnms2,
    save_split_audit,
    split_metadata,
)
from .distillation import (
    acdc_distillation_aggregate_results,
    acdc_distillation_train_one,
)
from .evaluate import (
    architecture_bank_calibrate_task,
    calibration_is_valid,
    reproduction_calibrate_task,
)
from .protocol import TASKS, TaskSpec, protocols, resolve_tasks
from .results import (
    acdc_2d_aggregate_results,
    acdc_3d_aggregate_results,
    aggregate_architecture_results,
    aggregate_cnn_results,
    aggregate_official_results,
    mnms2_3d_aggregate_results,
    mnms2_sax_2d_aggregate_results,
)
from .train import (
    acdc_2d_train_one_run,
    acdc_3d_train_one,
    mnms2_3d_train_one,
    mnms2_sax_2d_train_one_run,
    train_architecture,
    train_cnn,
)
from .utils import (
    acdc_3d_run_dir_for,
    acdc_3d_task_root,
    acdc_distillation_run_dir_for,
    acdc_distillation_task_root,
    checkpoint_root,
    cinema_git_commit,
    cleanup_cuda,
    mnms2_3d_run_dir_for,
    mnms2_3d_task_root,
    prepare_run,
    save_json,
)

ACDC_2D_CLASSES = protocols["acdc"]["classes"]


ACDC_2D_TASK_KEY = protocols["acdc_2d"]["task_key"]


ACDC_3D_CINEMA_RANDINIT_WEIGHT_DECAY = protocols["acdc_3d"][
    "cinema_randinit_weight_decay"
]


ACDC_3D_DATASET = protocols["acdc_3d"]["dataset"]


ACDC_3D_INITIALIZATION = protocols["shared"]["initialization"]


ACDC_3D_PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]


ACDC_3D_SHARED_ABLATION_MODELS = protocols["shared"]["shared_ablation_models"]


ACDC_3D_TASK_KEY = protocols["acdc_3d"]["task_key"]


ACDC_3D_VIEW = protocols["shared"]["view"]


ACDC_DISTILLATION_PROTOCOL_VERSION = protocols["acdc_distillation"]["protocol_version"]


MNMS2_3D_DATASET = protocols["mnms2_3d"]["dataset"]


MNMS2_3D_INITIALIZATION = protocols["shared"]["initialization"]


MNMS2_3D_SHARED_ABLATION_MODELS = protocols["shared"]["shared_ablation_models"]


MNMS2_3D_TASK_KEY = protocols["mnms2_3d"]["task_key"]


MNMS2_3D_VIEW = protocols["shared"]["view"]


MNMS2_SAX_2D_CLASSES = protocols["mnms2_sax_2d"]["classes"]


MNMS2_SAX_2D_TASK_KEY = protocols["mnms2_sax_2d"]["task_key"]


def acdc_2d_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages)

    print("=" * 100)
    print("ACDC patient-level 2-D architecture-bank benchmark")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Task: {ACDC_2D_TASK_KEY}")
    print(f"Classes: {list(ACDC_2D_CLASSES)}")
    print(f"Models: {list(args.models_2d)}")
    print(f"Formulations: {list(args.formulations_2d)}")
    print(f"Initializations: {list(args.initializations_2d)}")
    print(f"Seeds: {list(args.seeds)}")
    print(f"Stages: {list(args.stages)}")
    print("=" * 100)

    if "preprocess" in stages:
        preprocess_acdc(args)
    data_root = acdc_2d_resolve_processed_dir(args)
    splits = acdc_2d_load_official_splits(data_root)
    acdc_2d_save_split_audit(args.output_dir, splits)
    config = load_acdc_config(data_root, seed=0)
    if stages & {"validate", "train"}:
        acdc_2d_validate_inputs(args.output_dir, data_root, splits, config)

    if "train" in stages:
        for backbone in args.models_2d:
            for formulation in args.formulations_2d:
                for initialization in args.initializations_2d:
                    for seed in args.seeds:
                        acdc_2d_train_one_run(
                            args,
                            data_root,
                            splits,
                            backbone,
                            formulation,
                            initialization,
                            seed,
                        )
        if not args.smoke:
            acdc_2d_aggregate_results(args)
    print(f"Done. Results: {args.output_dir}")


def acdc_3d_save_run_manifest(args: argparse.Namespace) -> None:
    save_json(
        acdc_3d_task_root(args) / "run_manifest.json",
        {
            "protocol_version": ACDC_3D_PROTOCOL_VERSION,
            "dataset": ACDC_3D_DATASET,
            "task": ACDC_3D_TASK_KEY,
            "view": ACDC_3D_VIEW,
            "dimensionality": 3,
            "input_policy": "full_sax_ed_es_volume_per_patient",
            "models": list(args.models),
            "formulations": list(args.formulations),
            "shared_ablation_models": list(ACDC_3D_SHARED_ABLATION_MODELS),
            "initialization": ACDC_3D_INITIALIZATION,
            "optimizer_weight_decay": (
                ACDC_3D_CINEMA_RANDINIT_WEIGHT_DECAY
                if args.weight_decay is None
                else args.weight_decay
            ),
            "seeds": list(args.seeds),
            "stages": list(args.stages),
            "smoke": args.smoke,
            "resume": args.resume,
            "amp": args.amp,
            "deterministic": args.deterministic,
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cinema_commit": cinema_git_commit(),
        },
    )


def acdc_3d_record_failure(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
    error: BaseException,
) -> None:
    directory = acdc_3d_run_dir_for(args, backbone, formulation, seed)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(
        directory / "failure.json",
        {
            "protocol_version": ACDC_3D_PROTOCOL_VERSION,
            "dataset": ACDC_3D_DATASET,
            "task": ACDC_3D_TASK_KEY,
            "backbone": backbone,
            "formulation": formulation,
            "initialization": ACDC_3D_INITIALIZATION,
            "seed": seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )


def acdc_3d_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Seeds must be unique: {args.seeds}")

    print("=" * 100)
    print("ACDC FULL-VOLUME 3-D CNN ARCHITECTURE BANK")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Processed data: {acdc_3d_processed_dir(args)}")
    print(f"Models: {list(args.models)}")
    print(f"Formulations: {list(args.formulations)}")
    print(f"Seeds: {list(args.seeds)}")
    print("=" * 100)

    stages = set(args.stages)
    if "preprocess" in stages:
        acdc_3d_preprocess(args)
    if stages & {"validate", "train"}:
        acdc_3d_validate_inputs(args)
    if "train" in stages:
        acdc_3d_save_run_manifest(args)
        for backbone in args.models:
            for formulation in args.formulations:
                for seed in args.seeds:
                    try:
                        acdc_3d_train_one(args, backbone, formulation, seed)
                    except FileExistsError:
                        raise
                    except Exception as error:
                        acdc_3d_record_failure(args, backbone, formulation, seed, error)
                        cleanup_cuda()
                        print(
                            f"[failure] {backbone}/{formulation}/seed_{seed}: {type(error).__name__}: {error}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            raise
    if "aggregate" in stages or ("train" in stages and not args.smoke):
        acdc_3d_aggregate_results(args)
    print(f"Done. Results: {acdc_3d_task_root(args)}")


def acdc_distillation_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.temperature <= 0:
        raise ValueError("train.temperature must be positive")
    if not 0.0 <= args.supervised_weight <= 1.0:
        raise ValueError("train.supervised_weight must be between 0 and 1")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Seeds must be unique: {args.seeds}")

    stages = set(args.stages)
    if "preprocess" in stages:
        acdc_3d_preprocess(args)
    if stages & {"validate", "train"}:
        acdc_3d_validate_inputs(args)
    if "train" in stages:
        for backbone in args.models:
            for formulation in args.formulations:
                for seed in args.seeds:
                    try:
                        acdc_distillation_train_one(args, backbone, formulation, seed)
                    except FileExistsError:
                        raise
                    except Exception as error:
                        failure = (
                            acdc_distillation_run_dir_for(
                                args, backbone, formulation, seed
                            )
                            / "failure.json"
                        )
                        save_json(
                            failure,
                            {
                                "protocol_version": ACDC_DISTILLATION_PROTOCOL_VERSION,
                                "backbone": backbone,
                                "formulation": formulation,
                                "seed": seed,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        cleanup_cuda()
                        print(
                            f"[failure] {backbone}/{formulation}/seed_{seed}: {error}"
                        )
                        if not args.continue_on_error:
                            raise
    if "aggregate" in stages or ("train" in stages and not args.smoke):
        acdc_distillation_aggregate_results(args)
    gc.collect()
    print(f"Done. Distilled results: {acdc_distillation_task_root(args)}")


def architecture_bank_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages)
    run_2d = bool(stages & {"cnn", "cnn2d"})
    run_3d = "cnn3d" in stages
    selected = resolve_tasks(list(args.tasks))
    validation_specs = (
        list(selected) if stages & {"preprocess", "calibrate", "official"} else []
    )
    if run_2d:
        for key in args.cnn2d_tasks:
            if TASKS[key] not in validation_specs:
                validation_specs.append(TASKS[key])
    if run_3d:
        for key in args.cnn3d_tasks:
            if TASKS[key] not in validation_specs:
                validation_specs.append(TASKS[key])
    datasets = sorted({spec.dataset for spec in validation_specs})

    print("=" * 100)
    print("CineMA validation and 2-D/3-D architecture-bank benchmark")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Tasks: {[spec.key for spec in selected]}")
    if run_2d:
        print(f"2-D tasks/models: {list(args.cnn2d_tasks)} / {list(args.models_2d)}")
    if run_3d:
        print(f"3-D tasks/models: {list(args.cnn3d_tasks)} / {list(args.models_3d)}")
    print(f"Stages: {list(args.stages)}")
    print("=" * 100)

    if "preprocess" in args.stages:
        for dataset in datasets:
            preprocess_dataset(args, dataset)

    # Validate and persist splits before any evaluation or training.
    for spec in validation_specs:
        splits = split_metadata(spec, processed_dir(args, spec.dataset))
        save_split_audit(args, spec, splits)

    calibration_summaries = []
    if "calibrate" in args.stages:
        for spec in selected:
            calibration_summaries.append(architecture_bank_calibrate_task(args, spec))
        passed = all(item["passed"] for item in calibration_summaries)
        save_json(
            checkpoint_root(args) / "calibration" / "gate.json",
            {
                "passed": passed,
                "tasks": {x["task"]: x["passed"] for x in calibration_summaries},
            },
        )
        if not passed and not args.allow_calibration_failure:
            raise RuntimeError(
                "Released-checkpoint calibration failed. CNN/official training is intentionally blocked."
            )

    training_requested = "official" in stages or run_2d or run_3d
    if training_requested and not args.smoke and not args.allow_calibration_failure:
        gate_tasks: list[TaskSpec] = []
        if "official" in stages:
            gate_tasks.extend(selected)
        if run_2d:
            gate_tasks.extend(TASKS[key] for key in args.cnn2d_tasks)
        if run_3d:
            gate_tasks.extend(TASKS[key] for key in args.cnn3d_tasks)
        gate_tasks = list(dict.fromkeys(gate_tasks))
        if not calibration_is_valid(args, gate_tasks):
            raise RuntimeError(
                "Run run.stages=[preprocess,calibrate] first. Full training is blocked until released checkpoints reproduce."
            )

    if "official" in stages:
        for spec in selected:
            for mode in args.official_modes:
                for seed in args.seeds:
                    run_official_training(args, spec, mode, seed)
        aggregate_official_results(args)

    if run_2d:
        for task_key in args.cnn2d_tasks:
            spec = TASKS[task_key]
            for backbone in args.models_2d:
                for formulation in args.formulations_2d:
                    for initialization in args.initializations_2d:
                        for seed in args.seeds:
                            train_architecture(
                                args, spec, backbone, formulation, initialization, seed
                            )

    if run_3d:
        for task_key in args.cnn3d_tasks:
            spec = TASKS[task_key]
            for backbone in args.models_3d:
                for formulation in args.formulations_3d:
                    for seed in args.seeds:
                        train_architecture(
                            args, spec, backbone, formulation, "randinit", seed
                        )

    if (run_2d or run_3d) and not args.smoke:
        aggregate_architecture_results(args)

    print(f"Done. Results: {args.output_dir}")


def mnms2_3d_save_run_manifest(args: argparse.Namespace) -> None:
    save_json(
        mnms2_3d_task_root(args) / "run_manifest.json",
        {
            "dataset": MNMS2_3D_DATASET,
            "task": MNMS2_3D_TASK_KEY,
            "view": MNMS2_3D_VIEW,
            "dimensionality": 3,
            "input_policy": "full_sax_ed_es_volume_per_patient",
            "models": list(args.models),
            "formulations": list(args.formulations),
            "shared_ablation_models": list(MNMS2_3D_SHARED_ABLATION_MODELS),
            "initialization": MNMS2_3D_INITIALIZATION,
            "seeds": list(args.seeds),
            "stages": list(args.stages),
            "smoke": args.smoke,
            "resume": args.resume,
            "amp": args.amp,
            "deterministic": args.deterministic,
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cinema_commit": cinema_git_commit(),
        },
    )


def mnms2_3d_record_failure(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
    error: BaseException,
) -> None:
    directory = mnms2_3d_run_dir_for(args, backbone, formulation, seed)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(
        directory / "failure.json",
        {
            "dataset": MNMS2_3D_DATASET,
            "task": MNMS2_3D_TASK_KEY,
            "backbone": backbone,
            "formulation": formulation,
            "initialization": MNMS2_3D_INITIALIZATION,
            "seed": seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )


def mnms2_3d_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Seeds must be unique: {args.seeds}")

    print("=" * 100)
    print("M&Ms2 FULL-VOLUME 3-D CNN ARCHITECTURE BANK")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Processed data: {mnms2_3d_processed_dir(args)}")
    print(f"Models: {list(args.models)}")
    print(f"Formulations: {list(args.formulations)}")
    print(f"Seeds: {list(args.seeds)}")
    print("=" * 100)

    stages = set(args.stages)
    if "preprocess" in stages:
        mnms2_3d_preprocess(args)
    if stages & {"validate", "train"}:
        mnms2_3d_validate_inputs(args)
    if "train" in stages:
        mnms2_3d_save_run_manifest(args)
        for backbone in args.models:
            for formulation in args.formulations:
                for seed in args.seeds:
                    try:
                        mnms2_3d_train_one(args, backbone, formulation, seed)
                    except FileExistsError:
                        raise
                    except Exception as error:
                        mnms2_3d_record_failure(
                            args, backbone, formulation, seed, error
                        )
                        cleanup_cuda()
                        print(
                            f"[failure] {backbone}/{formulation}/seed_{seed}: {type(error).__name__}: {error}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            raise
    if "aggregate" in stages or ("train" in stages and not args.smoke):
        mnms2_3d_aggregate_results(args)
    print(f"Done. Results: {mnms2_3d_task_root(args)}")


def mnms2_sax_2d_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages)

    print("=" * 100)
    print("M&Ms2 central-SAX patient-level 2-D architecture-bank benchmark")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Task: {MNMS2_SAX_2D_TASK_KEY}")
    print(f"Classes: {list(MNMS2_SAX_2D_CLASSES)}")
    print(f"Models: {list(args.models_2d)}")
    print(f"Formulations: {list(args.formulations_2d)}")
    print(f"Initializations: {list(args.initializations_2d)}")
    print(f"Seeds: {list(args.seeds)}")
    print(f"Stages: {list(args.stages)}")
    print("=" * 100)

    if "preprocess" in stages:
        preprocess_mnms2(args)
    data_root = mnms2_sax_2d_resolve_processed_dir(args)
    splits = mnms2_sax_2d_load_official_splits(data_root)
    mnms2_sax_2d_save_split_audit(args.output_dir, splits)
    config = load_mnms2_config(data_root, seed=0)
    if stages & {"validate", "train"}:
        mnms2_sax_2d_validate_inputs(args.output_dir, data_root, splits, config)

    if "train" in stages:
        for backbone in args.models_2d:
            for formulation in args.formulations_2d:
                for initialization in args.initializations_2d:
                    for seed in args.seeds:
                        mnms2_sax_2d_train_one_run(
                            args,
                            data_root,
                            splits,
                            backbone,
                            formulation,
                            initialization,
                            seed,
                        )
        if not args.smoke:
            mnms2_sax_2d_aggregate_results(args)
    print(f"Done. Results: {args.output_dir}")


def reproduction_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = resolve_tasks(list(args.tasks))
    validation_specs = list(selected)
    if "cnn" in args.stages:
        for key in args.cnn_tasks:
            if TASKS[key] not in validation_specs:
                validation_specs.append(TASKS[key])
    datasets = sorted({spec.dataset for spec in validation_specs})

    print("=" * 100)
    print("CineMA exact reproduction and CNN-extension benchmark")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Tasks: {[spec.key for spec in selected]}")
    print(f"Stages: {list(args.stages)}")
    print("=" * 100)

    if "preprocess" in args.stages:
        for dataset in datasets:
            preprocess_dataset(args, dataset)

    # Validate and persist splits before any evaluation or training.
    for spec in validation_specs:
        splits = split_metadata(spec, processed_dir(args, spec.dataset))
        save_split_audit(args, spec, splits)

    calibration_summaries = []
    if "calibrate" in args.stages:
        for spec in selected:
            calibration_summaries.append(reproduction_calibrate_task(args, spec))
        passed = all(item["passed"] for item in calibration_summaries)
        save_json(
            checkpoint_root(args) / "calibration" / "gate.json",
            {
                "passed": passed,
                "tasks": {x["task"]: x["passed"] for x in calibration_summaries},
            },
        )
        if not passed and not args.allow_calibration_failure:
            raise RuntimeError(
                "Released-checkpoint calibration failed. CNN/official training is intentionally blocked."
            )

    training_requested = "official" in args.stages or "cnn" in args.stages
    if training_requested and not args.smoke and not args.allow_calibration_failure:
        gate_tasks = (
            selected
            if "official" in args.stages
            else [TASKS[k] for k in args.cnn_tasks]
        )
        if not calibration_is_valid(args, gate_tasks):
            raise RuntimeError(
                "Run run.stages=[preprocess,calibrate] first. Full training is blocked until released checkpoints reproduce."
            )

    if "official" in args.stages:
        for spec in selected:
            for mode in args.official_modes:
                for seed in args.seeds:
                    run_official_training(args, spec, mode, seed)
        aggregate_official_results(args)

    if "cnn" in args.stages:
        for task_key in args.cnn_tasks:
            spec = TASKS[task_key]
            for backbone in args.cnn_models:
                for formulation in args.formulations:
                    for initialization in args.initializations:
                        for seed in args.seeds:
                            train_cnn(
                                args, spec, backbone, formulation, initialization, seed
                            )
        aggregate_cnn_results(args)

    print(f"Done. Results: {args.output_dir}")


def run(config: DictConfig) -> None:
    """Validate the selected experiment and execute its requested stages."""
    experiment = str(config.experiment)
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {experiment!r}; choose from {list(EXPERIMENTS)}"
        )
    args = prepare_run(config, experiment)
    for category in ("checkpoints", "metrics", "predictions", "logs"):
        (args.output_dir / category).mkdir(parents=True, exist_ok=True)
    EXPERIMENTS[experiment](args)


EXPERIMENTS = {
    "acdc_2d": acdc_2d_run,
    "acdc_3d": acdc_3d_run,
    "acdc_distillation": acdc_distillation_run,
    "mnms2_sax_2d": mnms2_sax_2d_run,
    "mnms2_3d": mnms2_3d_run,
    "architecture_bank": architecture_bank_run,
    "reproduction": reproduction_run,
}
