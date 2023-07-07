import numpy as np

from sklearn.base import BaseEstimator
from typing import List

from tissue.data.loading_mean_baselines import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss


class Estimator(BaseEstimator):
    
    data: Data
    history: dict

    def __init__(self, 
                data: Data,
                model,
                monitor_partition: str = 'val',
    ):
        self.data = data
        self.model = model
        self.monitor_partition = monitor_partition
        
        
    
    def train(self):
        best_loss = float('inf')
        best_iteration = 0
        no_improvement_count = 0
        max_no_improvement = 5  # Maximum number of iterations without improvement
        self.train_loss = []
        self.val_loss = []

        for iteration in range(100):  # Set a maximum number of iterations if needed
            self.model.fit(self.data.X_train, self.data.y_train)
            if self.monitor_partition == 'val':
                val_predictions = self.model.predict_proba(self.data.X_val)
                validation_loss = log_loss(self.data.y_val, val_predictions)

                train_predictions = self.model.predict_proba(self.data.X_train)
                training_loss = log_loss(self.data.y_train, train_predictions)
                
                self.val_loss.append(validation_loss)
                self.train_loss.append(training_loss)

                print(f"iteration: {iteration}, val_loss={validation_loss}, train_loss={training_loss}")    

                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_iteration = iteration
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= max_no_improvement:
                    break
            else:
                train_predictions = self.model.predict_proba(self.data.X_train)
                training_loss = log_loss(self.data.y_train, train_predictions)                

                self.train_loss.append(training_loss)

                print(f"iteration: {iteration}, train_loss={training_loss}")    

                if training_loss < best_loss:
                    best_loss = training_loss
                    best_iteration = iteration
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= max_no_improvement:
                    break
        
            #if isinstance(self.model, RandomForestClassifier):
            #    self.model.set_params(n_estimators=best_iteration + 1)
        
    
        self.model.fit(self.data.X_train, self.data.y_train)
        self.history = {}
        self.history["train_loss"] = self.train_loss
        self.history["val_loss"] = self.val_loss

    def test(self):
        """
        return predictions on test data
        """
        return self.model.predict(self.data.X_test)
    
    def validate(self):
        """
        return predictions on val data
        """
        return self.model.predict(self.data.X_val)

    
    def predict(self, 
                idx: List[str]):
        """
        idx (List[str]): list of img keys
        """
        X = np.array([])
        for i in idx:
            if len(X) > 0:
                X = np.vstack((X, self.data.X[i]))
            else:
                X = np.array([self.data.X[i]])

        return self.model.predict_proba(X)
    