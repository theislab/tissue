import os
from typing import Dict, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins.io import AsyncCheckpointIO

import torch

from tissue.consts import BATCH_KEY_GRAPH_EMBEDDING, BATCH_KEY_NODE_EMBEDDING
from tissue.models.utils import move_to_numpy
from tissue.estimators.callbacks import HistoryCallback
from tissue.imports.datamodule import GraphAnnDataModule


class Estimator:
    datamodule: GraphAnnDataModule
    fn_ckpt: Union[None, str]
    history: dict
    model: pl.LightningModule
    trainer: pl.Trainer

    def __init__(self, datamodule: GraphAnnDataModule, model: pl.LightningModule):
        self.trainer = None
        self.datamodule = datamodule
        self.model = model
        self.fn_ckpt = None

    def _execution_checks(self):
        if not self.model:
            raise RuntimeError('You need to call self.init_model before calling self.train')
        if not self.datamodule:
            raise RuntimeError('You need to call self.init_datamodule before calling self.train')

    def prepare_trainer(
            self,
            epochs: int = 0,
            auto_lr_find: bool = False,
            callbacks: list = [],
            detect_anomaly: bool = False,
            enable_checkpointing: bool = False,
            enable_mps: bool = False,
            log_dir: Union[str, None] = None,
            max_steps_per_epoch: Union[int, None] = 20,
            max_validation_steps: Union[int, None] = 10,
            min_epochs: Union[int, None] = None,
            patience: int = 20,
            precision: int = 32,
            trainer_kwargs: Dict = {}
    ):
    
        self.train_hyperparam = {"epochs": epochs,
                                 "log_dir": log_dir,
                                 "max_steps_per_epoch": max_steps_per_epoch,
                                 "max_validation_steps": max_validation_steps,
                                 "patience": patience}
        trainer_kwargs_ = {
            # 'auto_lr_find': auto_lr_find,
            'accelerator': 'cpu',
            'check_val_every_n_epoch': 1,
            'detect_anomaly': detect_anomaly,
            'enable_checkpointing': enable_checkpointing,
            'limit_train_batches': max_steps_per_epoch,
            'limit_val_batches': max_validation_steps,
            'max_epochs': epochs,
            'min_epochs': min_epochs,
            'precision': precision
        }
        
        if not enable_mps and hasattr(torch.backends, "mps"):
            # Force model off of MPS, eg. when trying to avoid MPS specific errors.
            trainer_kwargs_["accelerator"] = "cpu"
            trainer_kwargs_["devices"] = "auto"
        else:
            trainer_kwargs_["accelerator"] = "auto"
            trainer_kwargs_["devices"] = "auto"
        if enable_checkpointing:
            trainer_kwargs_["plugins"] = [AsyncCheckpointIO()]
        trainer_kwargs_.update(trainer_kwargs)

        if log_dir is not None:
            tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "lightning_logs"))
            logger_kwargs = {"logger": tb_logger}
        else:
            logger_kwargs = {}
        self.trainer = pl.Trainer(callbacks=callbacks, **logger_kwargs, **trainer_kwargs_)

    def train(
            self,
            epochs: int,
            lr: Union[None, float],
            auto_lr_find: bool = False,
            cbs: list = [],
            checkpoint_every_n_train_steps: int = 1000,
            checkpoint_filename: Union[str, None] = None,
            detect_anomaly: bool = False,
            enable_checkpointing: bool = False,
            enable_mps: bool = False,
            l2: float = 0.,
            log_dir: Union[str, None] = None,
            lr_schedule_factor: float = 0.2,
            lr_schedule_min_lr: float = 1e-10,
            lr_schedule_patience: int = 10,
            max_steps_per_epoch: Union[int, None] = 20,
            max_validation_steps: Union[int, None] = 10,
            patience: int = 20,
            precision: int = 32,
            trainer_kwargs: Dict = {},
            monitor: str = "val_loss",
            **kwargs
    ):
        self._execution_checks()
        use_rich = True
        callbacks = [
            HistoryCallback(),
            pl.callbacks.EarlyStopping(monitor=monitor, mode='min', patience=patience),
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.ModelSummary(max_depth=3),
        ] + cbs
        # if use_rich:
        #     callbacks += [RichProgressBar()]
        if enable_checkpointing:
            assert log_dir is not None
            cb_ckpt = pl.callbacks.ModelCheckpoint(dirpath=log_dir,
                                                   every_n_train_steps=checkpoint_every_n_train_steps,
                                                   filename=checkpoint_filename, monitor=monitor, save_top_k=1)
            callbacks += [cb_ckpt]

        self.prepare_trainer(
            auto_lr_find=auto_lr_find,
            callbacks=callbacks,
            detect_anomaly=detect_anomaly,
            enable_checkpointing=enable_checkpointing,
            enable_mps=enable_mps,
            epochs=epochs,
            log_dir=log_dir,
            max_steps_per_epoch=max_steps_per_epoch,
            max_validation_steps=max_validation_steps,
            patience=patience,
            precision=precision,
            trainer_kwargs=trainer_kwargs)
        # Optimization hyper-parameters:
        # Regularization:
        self.model.l2_reg = l2
        # Learning rate:
        assert isinstance(lr, float), lr
        self.model.lr = lr
        self.model.lr_schedule_factor = lr_schedule_factor
        self.model.lr_schedule_min_lr = lr_schedule_min_lr
        self.model.lr_schedule_patience = lr_schedule_patience

        print("fitting model...")
        self.trainer.fit(self.model, datamodule=self.datamodule)
        if enable_checkpointing:
            # Report checkpoint path for future use, load best check-pointed parameters:
            self.fn_ckpt = cb_ckpt.best_model_path
            self.model.load_from_checkpoint(self.fn_ckpt)
        self.history = self._extract_history(trainer=self.trainer)

    def _extract_history(self, trainer):
        history = trainer.callbacks[0].metrics
        history = dict([(k, torch.stack(v, dim=0).cpu().numpy()) for k, v in history.items()])
        # validation histories are one too long because of validation sanity check before training (?)
        history = dict([(k, v[1:] if k.startswith("val") else v) for k, v in history.items()])
        return history

    def validate(self, **kwargs) -> dict:
        results = self.trainer.validate(
            model=self.model,
            dataloaders=self.datamodule,
            ckpt_path=None,
            verbose=False)[0]
        # Remove partition prefix from metric names:
        results = dict([("_".join(k.split("_")[1:]), v) for k, v in results.items()])
        return results

    def test(self, **kwargs) -> dict:
        results = self.trainer.test(
            model=self.model,
            dataloaders=self.datamodule,
            ckpt_path=None,
            verbose=False)[0]
        # Remove partition prefix from metric names:
        results = dict([("_".join(k.split("_")[1:]), v) for k, v in results.items()])
        return results
    
    def predict(self, idx, **kwargs) -> dict:
        results = self.trainer.predict(
            model=self.model,
            dataloaders=self.datamodule.predict_dataloader(idx=idx),
            return_predictions=True,
        )
        return results
        
    def graph_embedding(self, idx, **kwargs) -> np.ndarray:
        results = self.trainer.predict(
            model=self.model,
            dataloaders=self.datamodule.predict_dataloader(idx=idx),
            ckpt_path=None
        )
        embedding = np.vstack([move_to_numpy(x[BATCH_KEY_GRAPH_EMBEDDING]) for x in results])
        return embedding

    def node_embedding(self, idx, **kwargs) -> np.ndarray:
        results = self.trainer.predict(
            model=self.model,
            dataloaders=self.datamodule.predict_dataloader(idx=idx),
            ckpt_path=None
        )
        embedding = np.vstack([move_to_numpy(x[BATCH_KEY_NODE_EMBEDDING]) for x in results])
        return embedding

    def load_model_from_checkpoint(self, fn, cls):
        self.model = cls.load_from_checkpoint(fn)
