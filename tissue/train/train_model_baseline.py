import os
import pickle
import numpy as np
from typing import Union

from joblib import dump
from tissue.estimators.estimator_baseline import Estimator


def _try_save(fn, obj):
    with open(fn, 'wb') as f:
        pickle.dump(obj=obj, file=f)


class TrainModel:
    estimator: Estimator

    def init_estim(self, model, data, monitor_partition: str = "val"):
        self.estimator = Estimator(model=model, data=data, monitor_partition=monitor_partition)
        

    # def _save_evaluation(self, fn):
    #     img_keys = {
    #         'test': self.estimator.data.train_img_keys,
    #         'val': sself.estimator.data.train_img_keys,
    #         'train': self.estimator.data.train_img_keys,
    #     }
    #     evaluations = {}
    #     for partition, keys in img_keys.items():
    #         if len(keys) > 0:
    #             # TODO: implement evaluate
    #             evaluations[partition] = self.estimator.evaluate(keys)
    #         else:
    #             evaluations[partition] = None
    #     _try_save(fn + '_evaluation.pickle', evaluations)

    def _save_predictions(self, fn):
        img_keys = {
            'test': self.estimator.data.test_img_keys,
            'val': self.estimator.data.val_img_keys,
            'train': self.estimator.data.train_img_keys,
        }
        y_true = {
            'test': self.estimator.data.y_test,
            'val': self.estimator.data.y_val,
            'train': self.estimator.data.y_train,
        }
        predictions = {}
        for partition, keys in img_keys.items():
            if len(keys) > 0:
                predictions[partition] = {}
                predictions[partition]["y_hat"] = self.estimator.predict(keys)
                predictions[partition]["y_true"] = y_true[partition]
            else:
                predictions[partition] = None
                
        _try_save(fn + '_predictions.pickle', predictions)


    def _save_history(self, fn):
        _try_save(fn + "_history.pickle", self.estimator.history)

    # def _save_hyperparam(self, fn):
    #     _try_save(fn + "_hyperparam.pickle", self.estimator.train_hyperparam)
    

    def _save_model(
            self,
            fn,
    ):
        dump(self.estimator.model, fn + '_model.joblib')


    def save(self, fn):
        # self._save_get_data_args(fn=fn)
        self._save_model(fn=fn)
        # self._save_evaluation(fn=fn)
        self._save_predictions(fn=fn)
        self._save_history(fn=fn)
        # self._save_hyperparam(fn=fn)
        # self._save_data_info(fn=fn)