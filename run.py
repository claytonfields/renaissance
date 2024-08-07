import os
import copy
import pytorch_lightning as pl
import os
# os.environ["NCCL_DEBUG"] = "INFO"

from renaissance.config import ex
from renaissance.modules import RenaissanceTransformer
from renaissance.datamodules.multitask_datamodule import MTDataModule

import warnings
import torch

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

@ex.automain
def main(_config):
    
    # warnings.simplefilter("error")
    
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # print(_config)
    dm = MTDataModule(_config, dist=False)

    model = RenaissanceTransformer(_config)
    
    # Create name for directory to log results
    load_path = _config['load_path']
    exp_name = f'{_config["exp_name"]}'
    seed = _config['seed']
    
    def parse_load_path(load_path):
        drive, path_and_file = os.path.splitdrive(load_path)
        path, file = os.path.split(path_and_file)
        folders = []
        while True:
            path, folder = os.path.split(path)
        
            if folder != "":
                folders.append(folder)
            else:
                if path != "":
                    folders.append(path)
                break
        folders.reverse()
        result_dir = folders[-3]
        checkpoint_name = file.split("/")[-1][:-5]
        parsed_string = f"{result_dir}_{checkpoint_name}"
        return parsed_string
        
        
    if not load_path:
        image_size = _config['image_size']
        patch_size = _config['patch_size']
        batch_size = _config['batch_size']
        per_gpu_batchsize = _config['per_gpu_batchsize']
        train_steps = _config['max_steps']
        train_epoch = _config['max_epoch']
        result_dir = f"{exp_name}_seed{seed}_is{image_size}_ps{patch_size}_bs{batch_size}_pgbs{per_gpu_batchsize}_ts{train_steps}"
    else:
        loaded_model = parse_load_path(load_path)
        result_dir = f"{exp_name}_seed{seed}_from_{loaded_model}"
    
    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=result_dir
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    
    torch.set_float32_matmul_precision('medium')
    
    trainer = pl.Trainer(
        devices= _config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator = 'gpu',
        # strategy = 'ddp_notebook',
        strategy='ddp_find_unused_parameters_true',
        # strategy = 'ddp_spawn',
        # strategy='ddp',
        deterministic='warn',
        max_epochs=_config["max_epoch"], #if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        if _config["resume_from"]:
            trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        else:
            trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
    
    print()
    print('Results can be found in:')
    print('result/'+result_dir)
    print()
