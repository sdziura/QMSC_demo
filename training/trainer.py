import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import schedule

from config import FixedParams, ModelParams, NNParams, SVMParams, QNNParams


def get_trainer(fixed_params: FixedParams, tb_run_name: str):

    tb_logger = TensorBoardLogger("server/tb_logs/", name=tb_run_name)

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )

    if FixedParams.use_gpu:
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Configure PyTorch Profiler with a valid schedule
    if FixedParams.profiler_active_steps == 0:
        profiler = None
    else:
        profiler = PyTorchProfiler(
            schedule=schedule(
                wait=1,  # Number of warm-up steps
                warmup=1,  # Number of warm-up steps before recording
                active=fixed_params.profiler_active_steps,  # Number of steps to record
                repeat=2,  # Number of profiling cycles
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "server/tb_logs/profiler"
            ),
            profile_dataloader=True,  # Enable DataLoader profiling
        )

    trainer = pl.Trainer(
        max_epochs=fixed_params.max_epochs,
        accelerator=accelerator,
        val_check_interval=fixed_params.val_check_interval,
        logger=tb_logger,
        profiler=profiler,
        log_every_n_steps=1,  # to clear warning, in the model set to log every epoch only
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                filename="best_model",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            ),
            early_stopping,
        ],
    )
    return trainer
