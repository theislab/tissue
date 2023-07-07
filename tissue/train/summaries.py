import os
import pickle
from typing import Dict, List, Tuple, Union

import torch
import numpy as np
import pandas as pd

class GridSearchContainer:
    """
    Loads, processes and depicts the results of one grid search run.
    Exposes the following processed data objects as attributes to user:
        - summary_table: pandas Dataframe
            Summarise all metrics in a single table with rows for each runs and CV partition.
        - run_ids_clean: list
            List of IDs of completed runs (all files present).
        - idx_test:
            Dictionary with cross-validation as key and list of test indices as values.
        - idx_train:
            Dictionary with cross-validation as key and list of train indices as values.
        - idx_val:
            Dictionary with cross-validation as key and list of validation indices as values.
        - patient_idx_test:
            Dictionary with cross-validation as key and list of patient ids used for testing.
        - patient_idx_train:
            Dictionary with cross-validation as key and list of patient ids used for training.
        - patient_idx_val:
            Dictionary with cross-validation as key and list of patient ids used for validation.
        - patient_dict:
            Dictionary with cross-validation as key and a dictionary of indices to patient id as values.
            # TODO what has cross-validation to do with this? The single dictionarys for the different cross-validations
            # TODO all seem to be the same
        - evals:
            Contains losses and metrics for all labels for all runs, all cross-validation-splits, all partitions.
        - true_labels:
            For all cross-validation parts, all partitions the true one-hot encoded target labels.
        - runparams:
            A dictionary containing all hyperparams for all runs like label_selection, diseased_only, ...
        - info:
            Contains full results of a single selected run.
    """
    summary_table: pd.DataFrame
    run_ids_clean: List

    # idx_test: Dict
    # idx_train: Dict
    # idx_val: Dict
    # patient_idx_test: Dict
    # patient_idx_train: Dict
    # patient_idx_val: Dict

    patient_dict: dict
    metrics: dict
    true_labels: dict
    runparams: dict
    info: dict

    def __init__(
            self,
            source_path: Union[str, dict],
            gs_ids: List[str],
            multitask: bool = False,
    ):
        """
        :param source_path: Path as string or dictionary of paths by gs_ids.
        :param gs_ids: List of grid search IDs to load.
        """
        if isinstance(source_path, str):
            source_path = dict([(x, source_path) for x in gs_ids])
        self.source_path = source_path
        self.gs_ids = gs_ids
        self.multitask = multitask

    def load_gs(
            self,
            expected_files: list = [
                'predictions.pickle',
                'history.pickle',
                #'hyperparam.pickle',
                #'model_weights.pth',
            ],
            recompute_evaluation: bool = False,
    ):
        """
        Load all metrics from grid search output files.
        Core results are save in self.summary_table.
        :return:
        """
        self.summary_table = []
        self.runparams = {}
        self.run_ids_clean = {}
        self.evals = {}
        self.loss = {}

        for gs_id in self.gs_ids:
            # Collect runs that belong to grid search by looping over file names in directory.
            indir = self.source_path[gs_id] + gs_id + "/results/"
            # runs_ids are the unique hyper-parameter settings, which are again subsetted by cross-validation.
            # These ids are present in all files names but are only collected from the history files here.
            run_ids = np.sort(np.unique([
                "_".join(".".join(x.split(".")[:-1]).split("_")[:-1])
                for x in os.listdir(indir)
                if 'history' in x
            ]))
            

            cv_ids = np.sort(np.unique([  # identifiers of cross-validation splits
                "cv" + x.split("cv")[-1]
                for x in run_ids
            ]))
            
            run_ids = np.sort(np.unique([  # identifiers of hyper-parameters settings
                "_".join(x.split("cv")[0].split("_"))[:-1]
                for x in run_ids
            ]))

            run_ids_clean = []  # only IDs of completed runs (all files present)
            for r in run_ids:
                complete_run = True
                for cv in cv_ids:
                    # Check pickled files:
                    for end in expected_files:
                        fn = r + "_" + cv + "_" + end
                        if not os.path.isfile(indir + fn):
                            print("File %r missing" % fn)
                            complete_run = False
                # Check run parameter files (one per cross-validation set):
                fn = r + "_runparams.pickle"
                if not os.path.isfile(indir + fn):
                    print("File %r missing" % fn)
                    complete_run = False
                if not complete_run:
                    print("Run %r not successful" % (r + "_" + cv))
                    pass
                else:
                    run_ids_clean.append(r)

            # Load results and settings from completed runs:
            runparams = {}  # Dictionary over runs with model settings.
            for x in run_ids_clean:
                # Load model settings (these are shared across all partitions).
                if os.path.isfile(
                        indir + x + "_runparams.pickle"):  # TODO depreceate and only keep the first case (newer commit)
                    fn_runparams = indir + x + "_runparams.pickle"
                    with open(fn_runparams, 'rb') as f:
                        runparams[x] = pickle.load(f)
                else:
                    fn_runparams = indir + x + "_" + cv_ids[0] + "_runparams.pickle"
                    with open(fn_runparams, 'rb') as f:
                        runparams[x] = pickle.load(f)


                # load evaluation scores
                # print(indir + x + "_" + cv_ids[-1] + "_evaluations.pickle")
                if not recompute_evaluation and os.path.isfile(indir + x + "_" + cv_ids[-1] + "_evaluations.pickle"):
                    self.evals[x] = {}
                    for cv in cv_ids:
                        fn_evals = indir + x + "_" + cv + "_evaluations.pickle"

                        with open(fn_evals, 'rb') as f:
                            self.evals[x][cv] = pickle.load(f)
                else:
                    print('computing metrics...')
                    self.evals[x] = self._compute_evals(x, gs_id, cv_ids)
                    for cv in cv_ids:
                        fn_evals = indir + x + "_" + cv + "_evaluations.pickle"
                        with open(fn_evals, 'wb') as file:
                            pickle.dump(self.evals[x][cv], file)
                
                # load loss
                if os.path.isfile(indir + x + "_" + cv_ids[-1] + "_history.pickle"): 
                    self.loss[x] = {}
                    for cv in cv_ids:
                        fn_history = indir + x + "_" + cv + "_history.pickle"

                        with open(fn_history, 'rb') as f:
                            history = pickle.load(f)
                        self.loss[x][cv] = {}
                        for partition in ["train", "val", "test"]:
                            if f"{partition}_loss" in history.keys() and len(history[f"{partition}_loss"]) > 0:
                                self.loss[x][cv][partition] = history[f"{partition}_loss"][-1]
                            else:
                                self.loss[x][cv][partition] = None
                
            
                
            # Summarise all metrics in a single table with rows for each runs and CV partition.
            runparams[gs_id] = runparams
            self.run_ids_clean[gs_id] = run_ids_clean
            
            metrics = np.unique(np.concatenate([
                np.asarray(list(self.evals[x][cv_ids[0]]["train"].keys()))
                for x in run_ids_clean
            ])).tolist()
            
            self.summary_table.append(pd.concat([
            pd.DataFrame(dict(list({
                               "model_class": [str(runparams[x]['model_class']) for x in run_ids_clean]
                               if 'model_class' in list(runparams[x].keys()) else "none",
                                
                               "data_set": [str(runparams[x]['data_set']) for x in run_ids_clean]
                               if 'data_set' in list(runparams[x].keys()) else "none",
        
                               "radius": [str(runparams[x]['radius']) for x in run_ids_clean]
                               if 'radius' in list(runparams[x].keys()) else "none",
    
                               "target_label": [str(runparams[x]['target_label']) for x in run_ids_clean]
                               if 'target_label' in list(runparams[x].keys()) else "none",
        
                               "graph_label_selection": [str(runparams[x]['graph_label_selection']) for x in run_ids_clean]
                               if 'graph_label_selection' in list(runparams[x].keys()) else "none",
        
                               "feature_space": [str(runparams[x]['featur_space']) for x in run_ids_clean]
                               if 'featur_space' in list(runparams[x].keys()) else "none",
        
                               "learning_rate": [str(runparams[x]['learning_rate']) for x in run_ids_clean]
                               if 'learning_rate' in list(runparams[x].keys()) else "none",
        
                               "l2_reg": [str(runparams[x]['l2_reg']) for x in run_ids_clean]
                               if 'l2_reg' in list(runparams[x].keys()) else "none",
        
                               "batch_size": [str(runparams[x]['batch_size']) for x in run_ids_clean]
                               if 'batch_size' in list(runparams[x].keys()) else "none",
        
                               "graph_label_selection": [str(runparams[x]['graph_label_selection']) for
                                                           x in run_ids_clean]
                               if 'graph_label_selection' in list(runparams[x].keys()) else "none",
        
                               "depth_node_embedding": [str(runparams[x]['depth_node_embedding']) for x in
                                                     run_ids_clean]
                               if 'depth_node_embedding' in list(runparams[x].keys()) else "none",
        
                              "width_node_embedding": [str(runparams[x]['width_node_embedding']) for x in
                                                     run_ids_clean]
                               if 'width_node_embedding' in list(runparams[x].keys()) else "none",
        
                              "depth_graph_embedding": [str(runparams[x]['depth_graph_embedding']) for x in
                                                     run_ids_clean]
                               if 'depth_graph_embedding' in list(runparams[x].keys()) else "none",        
                            
                              "width_graph_embedding": [str(runparams[x]['width_graph_embedding']) for x in
                                                     run_ids_clean]
                               if 'width_graph_embedding' in list(runparams[x].keys()) else "none",     
        
                               "final_pooling": [str(runparams[x]['final_pooling']) for x in
                                                     run_ids_clean]
                               if 'final_pooling' in list(runparams[x].keys()) else "none",    
        
                               "multitask_setting": [str(runparams[x]['multitask_setting']) for x in
                                                     run_ids_clean]
                               if 'multitask_setting' in list(runparams[x].keys()) else "none",
        
                               "n_clusters": [str(runparams[x]['n_clusters']) for x in run_ids_clean]
                               if 'n_clusters' in list(runparams[x].keys()) else "none",
                               
                               "self_supervision_mode": [str(runparams[x]['self_supervision_mode']) for x in
                                                     run_ids_clean]
                               if 'self_supervision_mode' in list(runparams[x].keys()) else "none",
        
                               "node_supervision_mode": [str(runparams[x]['node_supervision_mode']) for x in
                                                     run_ids_clean]
                               if 'node_supervision_mode' in list(runparams[x].keys()) else "none",

                               "train_loss": [self.loss[x][cv]["train"] for x in run_ids_clean],
                               "test_loss": [self.loss[x][cv]["test"] for x in run_ids_clean],
                               "val_loss": [self.loss[x][cv]["val"] for x in run_ids_clean],
                               
                               "run_id": run_ids_clean,
                               "cv": cv,
                               "gs_id": gs_id
                           }.items()) +
                                  list(dict([
                                      ("train_" + m, [
                                          self.evals[x][cv]["train"][m] if m in self.evals[x][cv]["train"].keys()
                                          else np.nan for x in run_ids_clean
                                      ]) for m in metrics
                                  ]).items()) +
                                  list(dict([
                                      ("test_" + m, [
                                          self.evals[x][cv]["test"][m] if self.evals[x][cv]["test"] is not None and m in self.evals[x][cv]["test"].keys()
                                          else np.nan for x in run_ids_clean
                                      ]) for m in metrics
                                  ]).items()) +
                                  list(dict([
                                  ("val_" + m, [
                                          self.evals[x][cv]["val"][m] if self.evals[x][cv]["val"] is not None and m in self.evals[x][cv]["val"].keys()
                                          else np.nan for x in run_ids_clean
                                      ]) for m in metrics
                                  ]).items())
                                  )) for cv in cv_ids
            ]))

            # Summarise all metrics in a single table with rows for each runs and CV partition.
            if len(run_ids_clean) == 0:
                raise ValueError("no complete runs found")
            print("loaded %s: %i runs with %i-fold cross validation" %
                  (gs_id, len(self.run_ids_clean[gs_id]), len(cv_ids)))

        self.summary_table = pd.concat(self.summary_table)

        self.summary_table['model_type'] = [
            'graph' if 'gcn' in model or 'gin' in model  or model == 'spatial' else 'reference' for model in
            self.summary_table['model_class'].values
        ]
        

        self.summary_table['scenario'] = [
            'GNN' if c == 'gatinteraction' else
            'GNN' if c == 'gat' else
            'GNN' if c == 'gin' else
            'GNN' if c == 'gcn' else
            'GNN' if c == 'gcnss' else
            'S2' if c == 'mi' else
            'S2' if c == 'aggcond' else
            'S1' if c == 'reg' else
            'S3' if c == 'spatial' else
            'S1' if c == 'rfmean' else
            'S1' if c == 'regmean' else
            'S3' if c == 'rfdisp' else
            'S3' if c == 'regdisp' else
            'S3' if c == 'rfspatial' else
            'S3' if c  == 'regspatial' else
            c for c in self.summary_table["model_class"]
        ]

        self.summary_table["models"] = [
            'NN' if c == 'gatinteraction' else
            'NN' if c == 'gat' else
            'NN' if c == 'gin' else
            'NN' if c == 'gcn' else
            'NN' if c == 'gcnss' else
            'NN' if c == 'mi' else
            'NN' if c == 'aggcond' else
            'NN' if c == 'reg' else
            'NN' if c == 'spatial' else
            'RF' if c == 'rfmean' else
            'LogReg' if c == 'regmean' else
            'RF' if c == 'rfdisp' else
            'LogReg' if c == 'regdisp' else
            'RF' if c == 'rfspatial' else
            'LogReg' if c  == 'regspatial' else
            c for c in self.summary_table["model_class"]
        ]

        self.summary_table["model_groups"] = [
            'GAT' if c == 'gatinteraction' else
            'GAT' if c == 'gat' else
            'GIN' if c == 'gin' else
            'GCN' if c == 'gcn' else
            'GCNSS' if c == 'gcnss' else
            'MI' if c == 'mi' else
            'MEAN' if c == 'reg' else
            'AGG' if c == 'aggcond' else
            'GRAPH_SPATIAL' if c == 'spatial' else
            'MEAN' if c == 'rfmean' else
            'MEAN' if c == 'regmean' else
            'DISP' if c == 'rfdisp' else
            'DISP' if c == 'regdisp' else
            'DEGREE' if c == 'rfspatial' else
            'DEGREE' if c  == 'regspatial' else
            c for c in self.summary_table["model_class"]
        ]

        self.summary_table["model_class"] = [
            'GAT' if c == 'gatinteraction' else
            'GAT' if c == 'gat' else
            'GIN' if c == 'gin' else
            'GCN' if c == 'gcn' else
            'GCNSS' if c == 'gcnss' else
            'GCNSSO' if c == 'gcnsso' else
            'GSUM' if c == 'gsum' else
            'GAT' if c == 'gatcomplex' else
            'MI' if c == 'mi' else
            'MLP' if c == 'reg' else
            'AGGCOND' if c == 'aggcond' else
            'GRAPH_SPATIAL' if c == 'spatial' else
            'RF_MEAN' if c == 'rfmean' else
            'REG_MEAN' if c == 'regmean' else
            'RF_DISP' if c == 'rfdisp' else
            'REG_DISP' if c == 'regdisp' else
            'RF_SPATIAL' if c == 'rfspatial' else
            'REG_SPATIAL' if c  == 'regspatial' else
            c for c in self.summary_table["model_class"]
        ]

        

        self.summary_table['multitask'] = [
            'no' if setting == 'target' else 'yes' for setting in self.summary_table['multitask_setting'].values
        ]

        self.summary_table['test_cv'] = [int(e[2]) for e in self.summary_table['cv']]
        self.summary_table['val_cv'] = [int(e[4]) for e in self.summary_table['cv']]
        self.summary_table.drop('cv', axis=1, inplace=True)

    def get_best_model_id(
            self,
            subset_hyperparameters: List[Tuple[str, str]] = [],
            metric_select: str = "loss",
            partition_select: str = "val",
            cv_mode: str = "mean"
    ):
        """
        :param subset_hyperparameters:
        :param metric_select: Metric to use for ranking models.
        :param cv_mode: Summary mode of distribution of metric over partitions in cross validation. Out of:
            - "mean"
            - "median"
            - "max"
            - "min"
        :param partition_select: "train" or "eval" or "test" partition of data to collect metric from.
        :return:
        """
        if metric_select.endswith('acc') or \
                metric_select.endswith('accuracy') or metric_select.endswith('auc') or metric_select.endswith('aupr'):
            ascending = False
            if cv_mode == "min":
                raise Warning("selected cv_mode min with metric_id acc, likely not intended")
        elif metric_select.endswith('loss') or metric_select.endswith('crossentropy'):
            ascending = True
            if cv_mode == "max":
                raise Warning("selected cv_mode max with metric_id loss, likely not intended")
        elif metric_select.endswith('mse') or \
                metric_select.endswith('mae') or \
                metric_select.endswith('survival'):
            ascending = True
            if cv_mode == "max":
                raise Warning("selected cv_mode max with metric_id mae, likely not intended")
        else:
            raise ValueError("measure %s not recognized" % metric_select)

        if partition_select not in ["test", "val", "train"]:
            raise ValueError("partition %s not recognised" % partition_select)

        metric_select = partition_select + "_" + metric_select
        summary_table = self.summary_table.copy()
        for x, y in subset_hyperparameters:
            #if np.sum(summary_table[x].values == y) == 0:
            #    print(
            #        "subset was empty, available values for %s are %s, given was %s" %
            #        (x, str(np.unique(summary_table[x].values).tolist()), str(y))
            #    )
            summary_table = summary_table.loc[summary_table[x].values == y, :]
        if cv_mode.lower() == "mean":
            best_model = summary_table.groupby("run_id", as_index=False)[metric_select].mean(). \
                sort_values([metric_select], ascending=ascending)
        elif cv_mode.lower() == "median":
            best_model = summary_table.groupby("run_id", as_index=False)[metric_select].median(). \
                sort_values([metric_select], ascending=ascending)
        elif cv_mode.lower() == "max":
            best_model = summary_table.groupby("run_id", as_index=False)[metric_select].max(). \
                sort_values([metric_select], ascending=ascending)
        elif cv_mode.lower() == "min":
            best_model = summary_table.groupby("run_id", as_index=False)[metric_select].min(). \
                sort_values([metric_select], ascending=ascending)
        else:
            raise ValueError("cv_mode %s not recognized" % cv_mode)
        if best_model.shape[0] > 0:
            pass
            # print(metric_select + ": " + str(best_model[metric_select].values[0]))
        best_model_id = best_model['run_id'].values[0] if best_model.shape[0] > 0 else None
        best_val_cv = summary_table[summary_table["run_id"] == best_model_id]. \
            sort_values([metric_select], ascending=ascending)['val_cv'].values[0] if best_model_id is not None else None
        best_test_cv = summary_table[summary_table["run_id"] == best_model_id]. \
            sort_values([metric_select], ascending=ascending)['test_cv'].values[0] if best_model_id is not None else None
        best_gs_id = summary_table[summary_table["run_id"] == best_model_id]. \
            sort_values([metric_select], ascending=ascending)['gs_id'].values[0] if best_model_id is not None else None

        return best_gs_id, best_model_id, best_test_cv, best_val_cv
    
    def _compute_evals(
            self,
            model_id,
            gs_id,
            cv_ids,
            suffix="predictions",
    ):
        indir = self.source_path[gs_id] + gs_id + "/results/"
        for cv in cv_ids:
            fn = model_id + "_" + cv + "_" + suffix + ".pickle"
            if not os.path.isfile(indir + fn):
                raise ValueError("file %s missing" % suffix)
        info = {}
        for cv in cv_ids:
            info[cv] = {}

            fn = model_id + "_" + cv + "_" + suffix + ".pickle"
            with open(indir + fn, 'rb') as f:
                data = pickle.load(f)
                info[cv][suffix] = data
        
        if self.multitask:
            for cv in cv_ids:
                predictions = {}
                for partition in info[cv]["predictions"].keys():
                    predictions[partition] = {}
                    try:
                        for batch in info[cv]["predictions"][partition]:
                            keys = list(batch['graph_labels'].keys())
                            try:
                                predictions[partition]['graph_yhat'] = torch.concat((predictions[partition]['graph_yhat'], batch['graph_yhat'][0]))
                                keys = list(batch['graph_labels'].keys())
                                predictions[partition]['graph_labels'] = torch.concat((predictions[partition]['graph_labels'], batch['graph_labels'][keys[0]]))
                            except:
                                predictions[partition]['graph_yhat'] = batch['graph_yhat'][0]
                                predictions[partition]['graph_labels'] = batch['graph_labels'][keys[0]]
                    except:
                        predictions[partition] = None
                info[cv]["predictions"] = predictions   
                
        else:
            if not "y_hat" in info[cv_ids[0]]["predictions"]["train"][0].keys():
                for cv in cv_ids:
                    predictions = {}
                    for partition in info[cv]["predictions"].keys():
                        predictions[partition] = {}
                        try:

                            for part in info[cv]["predictions"][partition]:
                                for key in part.keys():
                                    try:
                                        predictions[partition][key] = torch.concat((predictions[partition][key], part[key]))
                                    except:
                                        predictions[partition][key] = part[key]
                        except:
                            predictions[partition] = None
                    info[cv]["predictions"] = predictions
            
        return self._compute_scores(info, cv_ids)
    
    def _compute_scores(
            self,
            info,
            cv_keys = None,
            partitions = ["train", "val", "test"],
        ):
        """
        Computes auroc, F1-score and Dice score
        """
        if cv_keys is None:
            cv_keys = list(info.keys())[:-1]

        scores = {}

        from torchmetrics import AUROC, Dice, AveragePrecision
        from torchmetrics.classification import MulticlassF1Score

        if "graph_yhat" in info[cv_keys[0]]["predictions"]["train"].keys():
            num_classes=info[cv_keys[0]]["predictions"]["train"]["graph_yhat"].shape[1]
        else: 
            num_classes=info[cv_keys[0]]["predictions"]["train"]["y_hat"].shape[1]
        for cv in cv_keys:
            scores[cv] = {}
            for partition in partitions:

                if info[cv]["predictions"][partition]:
                    scores[cv][partition] = {}
                    if "graph_yhat" in info[cv]["predictions"][partition]:
                        y_hat = torch.argmax(info[cv]["predictions"][partition]["graph_yhat"], dim=1)
                        preds = info[cv]["predictions"][partition]["graph_yhat"]

                        y_true = torch.argmax(info[cv]["predictions"][partition]["graph_labels"], dim=1)
                    else:
                        y_hat = torch.tensor(np.argmax(info[cv]["predictions"][partition]["y_hat"], axis=1))
                        preds = torch.tensor(info[cv]["predictions"][partition]["y_hat"])

                        y_true = torch.tensor(info[cv]["predictions"][partition]["y_true"], dtype=torch.long)

                    metric = AUROC(task="multiclass", num_classes=num_classes)
                    scores[cv][partition]["auc"] = metric(preds, y_true).item()

                    metric = MulticlassF1Score(num_classes=num_classes)
                    scores[cv][partition]["f1score"] = metric(y_hat, y_true).item()

                    metric = Dice(average='micro')
                    scores[cv][partition]["dice"] = metric(y_hat, y_true).item()

                    metric = AveragePrecision(task='multiclass', num_classes=num_classes)
                    scores[cv][partition]["aupr"] = metric(preds, y_true).item()
                else:
                    scores[cv][partition] = None
        return scores
    
    @property
    def cv_keys(self) -> List[str]:
        """
        Returns keys of cross-validation used in dictionaries in this class.
        :return: list of string keys
        """
        return np.unique(self.summary_table["cv"].values).tolist()

    def get_info(
            self,
            model_id,
            gs_id: str,
            expected_pickle=['history', 'hyperparam', 'predictions'],
            load_labels=["grade"]
    ):
        indir = self.source_path[gs_id] + gs_id + "/results/"
        # Check that all files are present:
        cv_ids = np.sort(np.unique([
            x[-20:-15] for x in os.listdir(indir)
            if x.split("_")[-1].split(".")[0] == "history"
        ]))
        for cv in cv_ids:
            # Check pickled files:
            for suffix in expected_pickle:
                fn = model_id + "_" + cv + "_" + suffix + ".pickle"
                if not os.path.isfile(indir + fn):
                    raise ValueError("file %s missing" % suffix)
        self.info = {}
        for cv in cv_ids:
            self.info[cv] = {}
            for suffix in expected_pickle:
                fn = model_id + "_" + cv + "_" + suffix + ".pickle"
                with open(indir + fn, 'rb') as f:
                    data = pickle.load(f)
                    self.info[cv][suffix] = data
        
        # concatenate batches in predictions
        if "predictions" in expected_pickle:
            import torch
            for cv in cv_ids:
                predictions = {}
                for partition in self.info[cv]["predictions"].keys():
                    predictions[partition] = {}
                    try:
                        for part in self.info[cv]["predictions"][partition]:
                            for key in part.keys():
                                try:
                                    predictions[partition][key] = torch.concat((predictions[partition][key], part[key]))
                                except:
                                    predictions[partition][key] = part[key]
                    except:
                        predictions[partition] = None
                self.info[cv]["predictions"] = predictions
                
        fn = model_id + "_runparams.pickle"
        with open(indir + fn, 'rb') as f:
            self.info["runparams"] = pickle.load(f)
    
    def _get_labels(
            self,
            cv_key,
            y_true_key: str = "graph_label",
            y_hat_key: str = "graph_yhat",
            partition_show: str = "test"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns observed labels and predicted labels.
        :param cv_idx: Index of cross-validation to plot confusion matrix for.
        :param y_true_key:
        :param y_hat_key:
        :param partition_show:
        :return: Tuple of predictions and labels. Each as numpy array
        """
        cv = cv_key
        
        predictions = self.info[cv]['predictions'][partition_show][y_hat_key]
        labels = self.info[cv]['predictions'][partition_show][y_true_key]
        return predictions, labels

    def _get_confusion_matrix(
            self,
            cv_key,
            y_true_key: str = "graph_label",
            y_hat_key: str = "graph_yhat",
            partition_show: str = "test",
            small: bool = False
    ) -> np.ndarray:
        """
        Returns confusion matrix of categorical prediction problem.
        :param cv_idx: Index of cross-validation to plot confusion matrix for.
        :param y_true_key:
        :param y_hat_key:
        :param partition_show:
        :return: Confusion matrix
        """
        import sklearn
        predictions, labels = self._get_labels(
            cv_key=cv_key,
            y_true_key=y_true_key,
            y_hat_key=y_hat_key,
            partition_show=partition_show
        )
        
        nr_labels = predictions.shape[1]
        labels = np.reshape(labels, predictions.shape)
        labels = np.argmax(labels, axis=1)
        predictions = np.argmax(predictions, axis=1)

        if small:
            l = range(1, nr_labels)
            labels[labels == 0] = 1
            predictions[predictions == 0] = 1
        else:
            l = range(0, nr_labels)

        return sklearn.metrics.confusion_matrix(
            y_true=labels,
            y_pred=predictions,
            labels=l
        )
    
    def plot_confusion_matrix(
            self,
            target_label: str = "grade",
            partition_show: str = "test",
            sum_across_cv: bool = False,
            mean_across_cv: bool = False,
            save: Union[str, None] = None,
            suffix: str = "_confusion_matrix.pdf",
            show: bool = True,
            return_axs: bool = False,
            small: bool = False,
            x_ticks=['1', '2', '3']
    ):
        """
        Plots the confusion matrix between the observed and predicted labels for all images.
        Use to visualise categorical predictors.
        :param metric_show:
        :param partition_show:
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :param small: For basel_zurich, whether to join grade 1 and 2.
        :return:
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        n_cv = len(self.info.keys()) - 1

        if not return_axs:
            plt.ioff()

        if sum_across_cv:
            fig, ax = plt.subplots(
                nrows=1, ncols=1,
                figsize=(3, 3)
            )
            res = []
            acc = []
            for i, cv_key in enumerate(list(self.info.keys())[:-1]):
                res_i = self._get_confusion_matrix(
                    cv_key=cv_key,
                    # target_label=target_label,
                    partition_show=partition_show,
                    small=small
                )
                res.append(res_i)
                acc.append(sum(res_i.diagonal()) / np.sum(res_i))
            res = sum(res)
            confusion_plot = ConfusionMatrixDisplay(res, display_labels=x_ticks)
            confusion_plot.plot(ax=ax)
            ax.set_title(", ".join([str(round(acc_i, 2)) for acc_i in acc]), y=1.15)
            ax.images[0].colorbar.remove()
        elif mean_across_cv:
            fig, ax = plt.subplots(
                nrows=1, ncols=1,
                figsize=(5, 5)
            )
            res = []
            acc = []
            for i, cv_key in enumerate(list(self.info.keys())[:-1]):
                res_i = self._get_confusion_matrix(
                    cv_key=cv_key,
                    # target_label=target_label,
                    partition_show=partition_show,
                    small=small
                )
                res.append(res_i)
                acc.append(sum(res_i.diagonal()) / np.sum(res_i))
            res = sum(res) / (i+1)
            confusion_plot = ConfusionMatrixDisplay(np.round(res, 2), display_labels=x_ticks)
            confusion_plot.plot(ax=ax)
            acc_mean = sum(res.diagonal()) / np.sum(res)
            ax.set_title(str(round(acc_mean, 2)), y=1.15)
            ax.images[0].colorbar.remove()
        else:
            fig, ax = plt.subplots(
                nrows=1, ncols=n_cv,
                figsize=(3 * n_cv, 3),
            )
            for i, cv_key in enumerate(list(self.info.keys())[:-1]):
                res = self._get_confusion_matrix(
                    cv_key=cv_key,
                    # target_label=target_label,
                    partition_show=partition_show,
                    small=small
                )
                acc = sum(res.diagonal()) / np.sum(res)
                confusion_plot = ConfusionMatrixDisplay(res, display_labels=x_ticks)
                confusion_plot.plot(ax=ax[i])
                ax[i].set_title(str(cv_key) + ": " + str(round(acc, 2)), y=1.15)
                ax[i].images[0].colorbar.remove()

        # Save, show and return figure.
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save is not None:
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(save + "_" + partition_show + suffix)
        if show:
            plt.grid(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
        if return_axs:
            return ax
        else:
            plt.close(fig)
            plt.ion()
            return None

    def plot_training_history(
            self,
            cv_keys: Union[str, None] = None,
            log_loss: bool = False,
            partition: str = "train",
            plot_lr=False,
            ada_scale: float = 0.1,
            panel_width=10.,
            panel_height=6.,
            rename_metrics: dict = {},
            save: Union[str, None] = None,
            suffix: str = "_training_history.pdf",
            show: bool = True,
            ax=None
    ):
        """
        Plot train and validation loss during training and learning rate reduction.
        :param cv_key: Index of cross-validation to plot training history for.
        :param log_loss:
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return:
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        if plot_lr:
            plts = 2
        else:
            plts = 1
        input_axs = ax is not None
        if not input_axs:
            plt.ioff()
            fig, ax = plt.subplots(
                nrows=1, ncols=plts,
                figsize=(panel_width * plts, panel_height)
            )
        if cv_keys is None:
            cv_keys = list(self.info.keys())[:-1]
        sns_data = []
        for cv in cv_keys:
            sns_data_temp = pd.DataFrame()
            sns_data_temp[f"loss"] = self.info[cv]["history"][f"{partition}_loss"]
            for k, v in sns_data_temp.items():
                sns_data_temp["epoch"] = np.arange(0, sns_data_temp.shape[0])
                sns_data_temp["cv test"] = cv.split('_')[0]
                sns_data_temp["cv val"] = 'cv' + cv.split('_')[1]
                sns_data.append(sns_data_temp)
        sns_data = pd.concat(sns_data, axis=0)

        
        if plot_lr:
            ax[0] = sns.lineplot(
                x="epoch", y="loss", hue="partition", style="cv",
                data=sns_data, ax=ax[0]
            )
            if log_loss:
                ax[0].set_ylabel("log loss")
            sns_data_lr = pd.DataFrame({
                "epoch": sns_data["epoch"].values,
                "cv": sns_data["cv"].values,
                "lr": np.log(sns_data["lr"].values) / np.log(10)
            })
            ax[1] = sns.lineplot(
                x="epoch", y="lr", style="cv",
                data=sns_data_lr, ax=ax[1]
            )
            ax[1].set_ylabel("log10 learning rate")
        else:
            ax = sns.lineplot(
                x="epoch", y="loss", hue="cv test", style="cv val",
                data=sns_data, ax=ax
            )
            if log_loss:
                ax.set_ylabel("log loss")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Save, show and return figure.
        if input_axs:
            pass
        else:
            plt.tight_layout()
            if save is not None:
                plt.tight_layout()
                plt.savefig(save + suffix)
            if show:
                plt.show()
            plt.close(fig)
            plt.ion()

    # def _compute_scores(
    #         self,
    #         score: str = "aucroc", # ["aucroc", "f1-score", "dice"]
    #         cv_keys: Union[str, None] = None,
    #         partition: str = "train",
    #         num_classes: int = 3,
    # ):
    #     if cv_keys is None:
    #         cv_keys = list(self.info.keys())[:-1]
        
    #     self.scores = {}
            
    #     from torchmetrics import PrecisionRecallCurve, AUROC, Dice
    #     from torchmetrics.classification import MulticlassF1Score

    #     if score.lower() == "aucroc":
    #         from torchmetrics import AUROC
    #         metric = AUROC(task="multiclass", num_classes=num_classes)

    #     elif "f1" in score.lower():
    #         from torchmetrics.classification import MulticlassF1Score
    #         metric = MulticlassF1Score(num_classes=num_classes)

    #     elif score.lower() == "dice":
    #         from torchmetrics import Dice
    #         metric = Dice(average='micro')
        
    #     else:
    #         raise ValueError(f"{score} is not implemented yet. Choose from ['aucroc', 'f1-score', 'dice']")

    #     for cv in cv_keys:
    #         self.scores[cv] = {}

    #         y_hat = torch.argmax(self.info[cv]["predictions"][partition]["graph_yhat"], dim=1)
    #         if "auc" in score:
    #             y_hat = self.info[cv]["predictions"][partition]["graph_yhat"]

    #         y_obs = torch.argmax(self.info[cv]["predictions"][partition]["graph_labels"], dim=1)
            

    #         self.score[cv] = metric(y_hat, y_obs)

    def plot_best_model_by_hyperparam(
            self,
            partition_show: str,
            metric_show: str,
            partition_select: str,
            metric_select: str,
            param_hue: str,
            param_x: str,
            cv_mode: Union[str, List[str]] = "mean",
            subset_hyperparam=[],
            show_swarm: bool = True,
            save: Union[str, None] = None,
            suffix: str = "_hyperparam.pdf",
            show: bool = True,
            return_axs: bool = False,
            panel_width: float = 5,
            panel_height: float = 3,
            xrot: float = 0,
            ttest: bool = False,
            param_scenario: str = "scenario",
            types: bool = False,
            dispersion: bool = False,
            feature_space: bool = False,
            multitask: bool = False,
            return_summary_table: bool = False,

    ):
        """
        Produces boxplots for all hyperparameters with multiple values. For every value for that hyperparameter
        the best model is chosen and all cvs are plotted.
        :param partition_show: "train" or "eval" or "test" partition of data.
        :param metric_show: Metric to plot.
        :param param_x: Hyper-parameter for x-axis partition.
        :param param_hue: Hyper-parameter for hue-axis partition.
        :param cv_mode: Summary mode of distribution of metric over partitions in cross validation. Out of:
            - "mean"
            - "median"
            - "max"
            - "min":param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :param panel_width:
        :param panel_height:
        :return:
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        params = [
            param for param in param_x
            if len(np.unique(self.summary_table[param])) > 1 and param != param_hue
        ]
        print("selected %s" % str(params))

        if isinstance(cv_mode, str):
            cv_mode = [cv_mode]

        plt.ioff()
        fig, ax = plt.subplots(
            nrows=len(params), ncols=len(cv_mode),
            figsize=(panel_width * len(cv_mode), panel_height * len(params)),
            sharey='col'
        )
        for i, param in enumerate(params):
            # Plot each metric:
            params_x_unique = np.sort(np.unique(self.summary_table[param].values))
            params_hue_unique = np.sort(np.unique(self.summary_table[param_hue].values))
            test_cvs = np.sort(np.unique(self.summary_table['test_cv'].values))
            for j, cvm in enumerate(cv_mode):
                run_ids = []
                test_cv_ids = []
                summary_table = self.summary_table.copy()
                for tcv in test_cvs:
                    for x in params_x_unique:
                        empty = True
                        for hue in params_hue_unique:
                            _, run_id_temp, _, _ = self.get_best_model_id(
                                subset_hyperparameters=[(param, x), (param_hue, hue),
                                                        ('test_cv', tcv)] + subset_hyperparam,
                                partition_select=partition_select,
                                metric_select=metric_select,
                                cv_mode=cvm
                            )
                            if run_id_temp is not None:
                                run_ids.append(run_id_temp)
                                test_cv_ids.append(tcv)
                                empty = False
                        if empty:
                            params_x_unique = [p for p in params_x_unique if p != x]
                if len(params_x_unique) > 1:
                    selected_runs = list(zip(run_ids, test_cv_ids))
                    subruns = [x in selected_runs for x in
                               zip(summary_table["run_id"].values, summary_table['test_cv'].values)]
                    summary_table = summary_table.loc[np.array(subruns), :].copy()
                    summary_table.sort_values([param, param_hue], inplace=True)
                    #                     summary_table[param_hue] = pd.Categorical(
                    #                         summary_table[param_hue].values,
                    #                         categories=np.sort(np.unique(summary_table[param_hue].values))
                    #                     )
                    ycol = partition_show + "_" + metric_show
                    if len(cv_mode) == 1 and len(params) == 1:
                        ax = np.array([ax])
                    if param == 'depth':
                        order = ['1', '2', '3', '5', '10']
                    elif param == 'node_feature_transformation':
                        order = ['none', 'standardize_globally', 'standardize_per_image', 'rank_per_image']
                    elif param == 'multitask_setting':
                        order = ['target', 'small', 'large']
                    elif param == 'l2':
                        order = ['0.0', '1e-06', '0.001', '1.0']
                    elif param == 'model_class':
                        order = ['RF_MEAN', 'REG_MEAN', 'MLP', 'AGGCOND', 'MI', 'RF_DISP', 'REG_DISP', 'GSUM', 'REG_SPATIAL', 'RF_SPATIAL', 'GRAPH_SPATIAL', 'GCN', 'GIN', 'GCNSS',
                                 'GCNADVANCED', 'GCNII', 'GAT', 'GATSINGLE', 'GATCOMPLEX']
                    elif param == "model_groups":
                        order = ["MEAN", "AGG", "MI", "DEGREE", "DISP", "GCN", "GIN", "GCNSS", "GAT"]

                    elif param == 'mmd':
                        order = ['0.0', '1.0', '5.0', '10.0']
                    elif param == 'adj_type':
                        order = ['scaled', 'spectral']
                    else:
                        order = None
                    if order is not None:
                        order = [o for o in order if o in params_x_unique]
                        for p in params_x_unique:
                            if p not in order:
                                print(f'There is no position for {p} defined in the ordering, so it is not plotted!')
                    if param_hue == "model_type":
                        hue_order = ["reference", "graph"]
                    elif param_hue == "models":
                        hue_order = ["LogReg", "RF", "NN"]
                    elif param_hue == "scenario":
                        hue_order = ["S1", "S2", "S3", "GNN"]
                    else:
                        hue_order = None

                    if return_summary_table:
                        ax = ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax
                        summary_table["partition_select"] = np.full(len(summary_table), partition_select)
                        return summary_table, param, ycol, order, ax
                    means = summary_table.groupby([param])[ycol].mean()
                    bp = sns.boxplot(
                        x=param, hue=param_hue, y=ycol,
                        order=order,
                        hue_order = hue_order,
                        data=summary_table, ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                        whis=1,
                    )

                    if ttest:
                        if feature_space:
                            from statannot import add_stat_annotation
                            add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                                                    ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                                                    box_pairs=[
                                                                (("MEAN", "celltype"), ("MEAN", "molecular")),
                                                                (("MI", "celltype"), ("MI", "molecular")),
                                                                (("GCN", "celltype"), ("GCN", "molecular")),
                                                                (("GCNSS", "celltype"), ("GCNSS", "molecular")),
                                                                (("GIN", "celltype"), ("GIN", "molecular")),
                                                                ],
                                                    order=order,
                                                    test='t-test_ind', text_format='star', loc='outside', verbose=2, comparisons_correction=None)
                        else:
                            ## copied from https://github.com/webermarcolivier/statannot/blob/master/statannot/statannot.py
                            def get_box_data(box_plotter, boxName):
                                """
                                boxName can be either a name "cat" or a tuple ("cat", "hue")

                                Here we really have to duplicate seaborn code, because there is not
                                direct access to the box_data in the BoxPlotter class.
                                """
                                cat = box_plotter.plot_hues is None and boxName or boxName[0]

                                index = box_plotter.group_names.index(cat)
                                group_data = box_plotter.plot_data[index]

                                if box_plotter.plot_hues is None:
                                    # Draw a single box or a set of boxes
                                    # with a single level of grouping
                                    box_data = remove_na(group_data)
                                else:
                                    hue_level = boxName[1]
                                    hue_mask = box_plotter.plot_hues[index] == hue_level
                                    box_data = remove_na(group_data[hue_mask])

                                return box_data
                            
                            ## get best models per scenario
                            from seaborn.utils import remove_na
                            box_plotter = sns.categorical._BoxPlotter(
                                        x=param, y=ycol, hue=param_hue, 
                                        order=order,
                                        hue_order=hue_order,
                                        data=summary_table, 
                                        orient=None, width=0.8, color=None,
                                        palette=None, saturation=.75, dodge=True, fliersize=5, linewidth=None)
                            

                            group_names = box_plotter.group_names
                            hue_names = box_plotter.hue_names

                            box_names = [(group_name, hue_name) for group_name in group_names for hue_name in hue_names]
                            labels = ['{}_{}'.format(group_name, hue_name) for (group_name, hue_name) in box_names]

                            box_structs = [{'box':box_names[i],
                                    'scenario': summary_table[summary_table[param] == box_names[i][0]][param_scenario].values[0],
                                    'label':labels[i],
                                    'box_data':get_box_data(box_plotter, box_names[i]),
                                    'ymean':np.mean(get_box_data(box_plotter, box_names[i])) if
                                        len(get_box_data(box_plotter, box_names[i])) > 0 else np.nan}
                                for i in range(len(box_names))]
                            
                            best_scenario = {}

                            for scenario in np.unique(summary_table[param_scenario]):
                                best_scenario[scenario] = {
                                    'model': None,
                                    'ymean': 0
                                }
                            
                            for box in box_structs:
                                if box['ymean'] > best_scenario[box['scenario']]['ymean']:
                                    best_scenario[box['scenario']]['ymean'] = box['ymean']
                                    best_scenario[box['scenario']]['model'] = box['box']
                                


                            from statannot import add_stat_annotation
                            add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                                                    ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                                                    box_pairs=[
                                                                (best_scenario["S1"]["model"], best_scenario["GNN"]["model"]),
                                                                (best_scenario["S2"]["model"], best_scenario["GNN"]["model"]),
                                                                (best_scenario["S3"]["model"], best_scenario["GNN"]["model"]),
                                                                (best_scenario["S1"]["model"], best_scenario["S2"]["model"]),
                                                                (("GCN", "NN"), ("GIN", "NN")),
                                                                ((best_scenario["S2"]["model"]), best_scenario["S3"]["model"])
                                                                ],
                                                    order=order,
                                                    test='t-test_ind', text_format='star', loc='outside', verbose=2, comparisons_correction=None)

                    # if ttest:
                    #     from statannot import add_stat_annotation
                    #     if feature_space:
                    #         add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                    #                         ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                    #                         box_pairs=[
                    #                                     (("MLP", "types"), ("MLP", "molecular")),
                    #                                     (("MI", "types"), ("MI", "molecular")),
                    #                                     (("GCN", "types"), ("GCN", "molecular")),
                    #                                     (("GCNSS", "types"), ("GCNSS", "molecular")),
                    #                                     ],
                    #                         order=order,
                    #                         test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                    #     elif types:
                    #         if dispersion:
                    #                 add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                    #                             ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                    #                             box_pairs=[
                    #                                         (("MLP", "types"), ("MI", "types")),
                    #                                         (("MI", "types"), ("GCN", "types")),
                    #                                         (("MI", "types"), ("GCNSS", "types")),
                    #                                         (("MI", "types"), ("DISP", "types")),
                    #                                         ],
                    #                             order=order,
                    #                             test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                    #         else:
                    #             add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                    #                             ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                    #                             box_pairs=[
                    #                                         (("MLP", "types"), ("MI", "types")),
                    #                                         (("MI", "types"), ("GCN", "types")),
                    #                                         (("MI", "types"), ("GCNSS", "types")),
                    #                                         ],
                    #                             order=order,
                    #                             test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                    #     elif multitask:
                    #         add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                    #                         ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                    #                         box_pairs=[
                    #                                     (("MLP", "target"), ("MLP", "small")),
                    #                                     (("MI", "target"), ("MI", "small")),
                    #                                     (("GCN", "target"), ("GCN", "small")),
                    #                                     (("GCNSS", "target"), ("GCNSS", "small")),
                    #                                     ],
                    #                         order=order,
                    #                         test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                    #     else:
                    #         add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                    #                             ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                    #                             box_pairs=[
                    #                                         (("MLP", "molecular"), ("MI", "molecular")),
                    #                                         (("MI", "molecular"), ("GCN", "molecular")),
                    #                                         (("MI", "molecular"), ("GCNSS", "molecular")),
                    #                                         #(("MLP", "molecular"), ("GCNSS", "molecular")),
                    #                                         #(("GCN", "molecular"), ("GCNSS", "molecular")),
                    #                                         ],
                    #                             order=order,
                    #                             test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                    handles, labels = bp.get_legend_handles_labels()
                    if show_swarm:
                        sns.swarmplot(
                            x=param, hue=param_hue, y=ycol, order=order,
                            data=summary_table,
                            ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                            # palette=[sns.color_palette("gray_r")[0], sns.color_palette("gray_r")[1]]
                            palette=['black']
                        )
                    if 'acc' in metric_show:
                        y_label = partition_show + ' accuracy'
                    elif 'mae' in metric_show:
                        y_label = partition_show + ' mae'
                    else:
                        y_label = ycol
                    ax[i].set_ylabel(y_label, rotation=90)
                    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=xrot)
                    box = ax[i].get_position()
                    ax[i].set_position([box.x0, box.y0, box.width * 0.95, box.height])
                    ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handles, labels=labels,
                                 title=param_hue)
        plt.ylim(bottom=0.0, top=1.1)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        #plt.xticks(rotation=90)

        # if param_x == ['model_class']:
        #     plt.legend(bbox_to_anchor=(1., 0.9), title=param_hue)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.tight_layout()
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None