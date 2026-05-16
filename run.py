"""
Renaissance training / evaluation entry point.

Usage:
    python run.py configs/pretrain_two_tower.yaml
    python run.py configs/pretrain_two_tower.yaml training.max_steps=50000
    python run.py configs/finetune_vqa.yaml experiment.test_only=true
"""

import os
import sys
import torch
from omegaconf import OmegaConf

from renaissance.config_schema import from_omegaconf
from renaissance.modeling import RenaissanceModel
from renaissance.trainer import RenaissanceTrainer


def _build_log_dir(cfg: dict) -> str:
    load_path = cfg.get("load_path", "")
    exp_name = cfg["exp_name"]
    seed = cfg.get("seed", 0)
    if not load_path:
        return (
            f"{exp_name}_seed{seed}"
            f"_is{cfg['image_size']}_ps{cfg['patch_size']}"
            f"_bs{cfg['batch_size']}_pgbs{cfg['per_gpu_batchsize']}"
            f"_ts{cfg['max_steps']}"
        )
    ckpt_name = os.path.splitext(os.path.basename(load_path))[0]
    return f"{exp_name}_seed{seed}_from_{ckpt_name}"


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print("Usage: python run.py <config.yaml> [key=value ...]", file=sys.stderr)
        sys.exit(1)

    config_path, overrides = args[0], args[1:]

    base_cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_dotlist(overrides) if overrides else OmegaConf.create({})
    omega_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    _config = from_omegaconf(omega_cfg)

    seed = _config.get("seed", 0)
    torch.manual_seed(seed)

    print("\n\nRunning Renaissance vision-language platform", file=sys.stderr)
    print(f"Task: {_config['exp_name']}", file=sys.stderr)
    print(f"Model type: {_config['model_type']}", file=sys.stderr)
    if _config["model_type"] == "one-tower":
        print(f"  encoder: {_config['encoder']}", file=sys.stderr)
    elif _config["model_type"] == "two-tower":
        print(f"  image_encoder: {_config['image_encoder']}", file=sys.stderr)
        print(f"  text_encoder:  {_config['text_encoder']}", file=sys.stderr)
    print(f"  lr={_config['learning_rate']}  max_steps={_config['max_steps']}\n\n", file=sys.stderr)

    result_dir = _build_log_dir(_config)
    log_dir = os.path.join(_config["log_dir"], result_dir)
    _config["log_dir"] = log_dir

    backend = _config.get("backend", "legacy")
    test_only = _config.get("test_only", False)

    if backend == "modern":
        from renaissance.data.runner import build_dataloader

        train_loader = None if test_only else build_dataloader(_config, split="train")
        val_loader = build_dataloader(_config, split="test" if test_only else "val")
    elif backend == "legacy":
        from renaissance.datamodules.multitask_datamodule import MTDataModule

        dm = MTDataModule(_config, dist=True)
        dm.setup("fit")
        train_loader = None if test_only else dm.train_dataloader()
        val_loader = dm.test_dataloader() if test_only else dm.val_dataloader()
    else:
        raise ValueError(f"data.backend must be 'legacy' or 'modern', got {backend!r}")

    model = RenaissanceModel(_config)

    if not test_only:
        trainer = RenaissanceTrainer(
            model, _config,
            train_dataloader=train_loader, val_dataloader=val_loader,
        )
        trainer.fit()
        print(f"\nResults in: {log_dir}\n")
    else:
        trainer = RenaissanceTrainer(model, _config, train_dataloader=val_loader)
        trainer.test()


if __name__ == "__main__":
    main()
