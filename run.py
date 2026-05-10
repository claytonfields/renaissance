import os
import copy
import sys
import torch

from renaissance.config import ex
from renaissance.modules import RenaissanceTransformer
from renaissance.datamodules.multitask_datamodule import MTDataModule
from renaissance.trainer import RenaissanceTrainer


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    if hasattr(torch, "manual_seed") and "seed" in _config:
        torch.manual_seed(_config["seed"])

    dm = MTDataModule(_config, dist=True)
    model = RenaissanceTransformer(_config)

    load_path = _config["load_path"]
    exp_name = _config["exp_name"]
    seed = _config.get("seed", 0)

    print("\n\nRunning Renaissance vision-language platform", file=sys.stderr)
    print(f"Task: {exp_name}", file=sys.stderr)
    print(f"Model type: {_config['model_type']}", file=sys.stderr)
    if _config["model_type"] == "one-tower":
        print(f"  encoder: {_config['encoder']}", file=sys.stderr)
    elif _config["model_type"] == "two-tower":
        print(f"  image_encoder: {_config['image_encoder']}", file=sys.stderr)
        print(f"  text_encoder:  {_config['text_encoder']}", file=sys.stderr)
    print(f"  lr={_config['learning_rate']}  max_steps={_config['max_steps']}\n\n", file=sys.stderr)

    if not load_path:
        result_dir = (
            f"{exp_name}_seed{seed}"
            f"_is{_config['image_size']}_ps{_config['patch_size']}"
            f"_bs{_config['batch_size']}_pgbs{_config['per_gpu_batchsize']}"
            f"_ts{_config['max_steps']}"
        )
    else:
        ckpt_name = os.path.splitext(os.path.basename(load_path))[0]
        result_dir = f"{exp_name}_seed{seed}_from_{ckpt_name}"

    log_dir = os.path.join(_config["log_dir"], result_dir)
    _config["log_dir"] = log_dir

    dm.setup("fit")

    if not _config["test_only"]:
        trainer = RenaissanceTrainer(
            model,
            _config,
            train_dataloader=dm.train_dataloader(),
            val_dataloader=dm.val_dataloader(),
        )
        trainer.fit()
        print(f"\nResults in: {log_dir}\n")
    else:
        trainer = RenaissanceTrainer(
            model,
            _config,
            train_dataloader=dm.test_dataloader(),
        )
        trainer.test()
