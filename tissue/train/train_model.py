import pickle

import torch
from tissue.estimators.base import Estimator


def _try_save(fn, obj):
    with open(fn, 'wb') as f:
        pickle.dump(obj=obj, file=f)


class TrainModel:
    estimator: Estimator

    def init_estim(self, model, datamodule):
        self.estimator = Estimator(model=model, datamodule=datamodule)

    def _save_evaluation(self, fn):
        img_keys = {
            'test': self.estimator.datamodule.idx_test,
            'val': self.estimator.datamodule.idx_val,
            'train': self.estimator.datamodule.idx_train,
        }
        evaluations = {}
        for partition, keys in img_keys.items():
            if len(keys) > 0:
                # TODO: implement evaluate
                evaluations[partition] = self.estimator.evaluate(keys)
            else:
                evaluations[partition] = None
        _try_save(fn + '_evaluation.pickle', evaluations)

    def _save_predictions(self, fn):
        img_keys = {
            'test': self.estimator.datamodule.idx_test,
            'val': self.estimator.datamodule.idx_val,
            'train': self.estimator.datamodule.idx_train,
        }

        predictions = {}
        for partition, keys in img_keys.items():
            if len(keys) > 0:
                predictions[partition] = self.estimator.predict(keys)
            else:
                predictions[partition] = None
        
        _try_save(fn + '_img_keys.pickle', img_keys)
        _try_save(fn + '_predictions.pickle', predictions)

    """def _save_data_info(self, fn):
         true_test = {name: [] for name in self.estimator.graph_label_selection}
         true_train = {name: [] for name in self.estimator.graph_label_selection}
         true_val = {name: [] for name in self.estimator.graph_label_selection}
         for ind in self.estimator.img_keys_test:
             for i, name in enumerate(self.estimator.graph_label_selection):
                 true_test[name].append(np.expand_dims(self.estimator.y[ind][name], axis=0))
         for ind in self.estimator.img_keys_eval:
             for i, name in enumerate(self.estimator.graph_label_selection):
                 true_val[name].append(np.expand_dims(self.estimator.y[ind][name], axis=0))
         for ind in self.estimator.img_keys_train:
             for i, name in enumerate(self.estimator.graph_label_selection):
                 true_train[name].append(np.expand_dims(self.estimator.y[ind][name], axis=0))
         true_values = {
             'test': true_test,
             'val': true_val,
             'train': true_train
         }
         label_transformations = {
             "continuous_mean": self.estimator.data.celldata.uns["graph_covariates"]["continuous_mean"],
             "continuous_std": self.estimator.data.celldata.uns["graph_covariates"]["continuous_std"],
             # "survival_mean": self.estimator.data.celldata.uns["graph_covariates"].survival_mean 
         }
         info = {
             'patient_dict': self.estimator.img_to_patient_dict,
             'true_targets': true_values,
             'label_transformations': label_transformations
         }
         _try_save(fn + "_datainfo.pickle", info)"""

    def _save_history(self, fn):
        _try_save(fn + "_history.pickle", self.estimator.history)

    def _save_hyperparam(self, fn):
        _try_save(fn + "_hyperparam.pickle", self.estimator.train_hyperparam)

    def _save_model(
            self,
            fn,
            save_weights: bool = True
    ):
        if save_weights:
            torch.save(self.estimator.model.state_dict(), fn + '_model_weights.pth')
        torch.save(self.estimator.model, fn + '_model.pth')

    def save(self, fn, save_weights: bool = False):
        # self._save_get_data_args(fn=fn)
        self._save_model(fn=fn, save_weights=save_weights)
        # self._save_evaluation(fn=fn)
        self._save_predictions(fn=fn)
        self._save_history(fn=fn)
        self._save_hyperparam(fn=fn)
        # self._save_data_info(fn=fn)
