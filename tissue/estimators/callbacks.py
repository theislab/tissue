from pytorch_lightning import callbacks


class HistoryCallback(callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def on_train_epoch_end(self, trainer, pl_module):
        x = trainer.callback_metrics
        for k, v in x.items():
            if k.startswith("train"):
                if k in self.metrics.keys():
                    self.metrics[k].append(v)
                else:
                    self.metrics[k] = [v]

    def on_validation_epoch_end(self, trainer, pl_module):
        x = trainer.callback_metrics
        for k, v in x.items():
            if k.startswith("val"):
                if k in self.metrics.keys():
                    self.metrics[k].append(v)
                else:
                    self.metrics[k] = [v]
