from typing import List, Union

import anndata as ad
import numpy as np

from tissue.consts import BATCH_KEY_GRAPH_LABELS, BATCH_KEY_LOCAL_ASSIGNMENTS, BATCH_KEY_LOCAL_LABELS, \
    BATCH_KEY_NODE_FEATURES, BATCH_KEY_NODE_LABELS
from tissue.imports.anndata2data import AnnData2DataDefault
from tissue.imports.datamodule import GraphAnnDataModule

from tissue.data.datasets import DatasetBaselZurichZenodo, DatasetMetabric, DatasetSchuerch
from tissue.data.clustering import get_self_supervision_label


def get_datamodule_from_adata(
        adatas: List[ad.AnnData],
        key_x: str,
        key_graph_supervision: str,
        batch_size: int = 1,
        edge_index: bool = False,
        key_local_assignment: Union[None, str] = None,
        key_local_supervision: Union[None, str] = None,
        key_node_supervision: Union[None, str] = None,
        num_workers: int = 1,
        preprocess=None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        idx_train: List[int] = None,
        idx_test: List[int] = None,
        idx_val: List[int] = None,
        seed: int = 42,
) -> GraphAnnDataModule:
    """
    Args:
    adatas (anndata.AnnData): Full dataset.
    keys_x (List[str]): List of node features and node labels fields to be stored in torch_geometric.Data.x
    keys_y (List[str]): List of graph level labels to be stored in torch_geometric.Data.y
    key_sample (str): adata obs field used to iterate over adata, usually image_id or patient_id
    edge_index (bool, optional): whether to set edge_index in torch_geometric.Data or not
        ToDo: False not supported right now.
    preprocess (List, optional): preprocessing functions to be applied
        Defaults to None
    self_supervision_label (List[str], optional): list of self supervision task lables
        Defaults to None
    n_clusters (int, optional): number of spectral clustering used for self supervision task.
        Defaults to 5

    Returns: Instance of GraphAnnDataModule.
    """
    assert (key_local_assignment is not None and key_local_supervision is not None) or \
           (key_local_assignment is None and key_local_assignment is None), \
        (key_local_assignment, key_local_supervision)
    if not edge_index:
        print("WARNING: edge_index==False not supported yet.")
    fields_x = {BATCH_KEY_NODE_FEATURES: key_x}
    fields_y = {BATCH_KEY_GRAPH_LABELS: key_graph_supervision}
    if key_local_assignment is not None:
        fields_x[BATCH_KEY_LOCAL_ASSIGNMENTS] = key_local_assignment
        fields_y[BATCH_KEY_LOCAL_LABELS] = key_local_supervision
    if key_node_supervision is not None:
        fields_x[BATCH_KEY_NODE_LABELS] = key_node_supervision
    converter = AnnData2DataDefault(
        adata_iter=lambda x: x,
        fields_x=fields_x,
        fields_y=fields_y,
        preprocess=preprocess,
        yields_edge_index=edge_index)
    
    dm = GraphAnnDataModule(
        datas=converter(adatas),
        batch_size=batch_size,
        num_workers=num_workers,
        learning_type="graph",
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        idx_train=idx_train,
        idx_test=idx_test,
        idx_val=idx_val,
        )
    if key_x.startswith("obsm/"):
        dm.dim_features = adatas[0].obsm[key_x.split("obsm/")[1]].shape[1]
    elif key_x.startswith("layers/"):
        dm.dim_features = adatas[0].layers[key_x.split("layers/")[1]].shape[1]
    elif key_x == "X":
        dm.dim_features = adatas[0].X.shape[1]
    else:
        assert False

    dm.dim_node_label = None
    if key_node_supervision:
        if key_node_supervision.startswith("obsm/"):
            dm.dim_node_label = adatas[0].obsm[key_node_supervision.split("obsm/")[1]].shape[1]
        elif key_node_supervision.startswith("layers/"):
            dm.dim_node_label = adatas[0].layers[key_node_supervision.split("layers/")[1]].shape[1]
        else:
            assert False
   
    return dm


def get_datamodule_from_curated(
        dataset: str,
        data_path: str, 
        radius: int,
        
        key_x: str,
        key_graph_supervision: str,
        batch_size: int = 1,
        
        buffered_data_path: str = None,
        cell_type_coarseness: str = "fine", 
        
        edge_index: bool = False,
        key_local_assignment: Union[None, str] = None,
        key_local_supervision: Union[None, str] = None,
        key_node_supervision: Union[None, str] = None,
        num_workers: int = 0,
        preprocess=None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        n_clusters: int = 5,

        seed_test: int = 42,
        seed_val: int = 42,
        **kwargs
) -> GraphAnnDataModule:
    """
    Extract objects from these curated datasets here and passes them to GraphDataset.
    """
    # TODO: load curated dataset here based on name of dataset and extrac adatas
    if dataset.lower() == "schuerch":
        Dataset = DatasetSchuerch
    elif dataset.lower() == "metabric":
        Dataset = DatasetMetabric
    elif dataset.lower() in ["jackson", "baselzurich", "zenodo"]:
        Dataset = DatasetBaselZurichZenodo
    else:
        raise ValueError("Dataset %r is not curated yet." % dataset)
    adata = Dataset(
        data_path=data_path,
        buffered_data_path=buffered_data_path,
        radius=radius,
        cell_type_coarseness=cell_type_coarseness
    )

    if dataset.lower() == 'metabric':  # filter out graphs without grade label
        exclude_indices = []
        img_celldata = {}
        img_to_patient_dict = {}
        for i, a in adata.img_celldata.items():
            if np.isnan(a.uns["graph_covariates"]["label_tensors"]["grade"]).any():
                exclude_indices.append(int(i))
            else:
                img_celldata[i] = a
                img_to_patient_dict[i] = adata.img_to_patient_dict[i]
        celldata = adata.celldata[~adata.celldata.obs["ImageNumber"].isin(exclude_indices)]
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        celldata.uns["img_keys"] = list(set(celldata.uns["img_keys"]) - set(exclude_indices))
        patients = np.array(list(img_to_patient_dict.values()))
        patients.sort()

        adata.img_celldata = img_celldata
        adata.celldata = celldata
        adata.img_to_patient_dict = img_to_patient_dict

    idx_train = None
    idx_test = None
    idx_val = None


    if isinstance(key_graph_supervision, list):
        # drop images with nans in any graph level
        exclude_indices = []
        img_celldata = {}
        img_to_patient_dict = {}
        for i, a in adata.img_celldata.items():
            keep = True
            for key in key_graph_supervision:
                key = key.split("/")[-1]
                if np.isnan(a.uns["graph_covariates"]["label_tensors"][key]).any():
                        keep = False
                        break
            if keep:
                img_celldata[i] = a
                img_to_patient_dict[i] = adata.img_to_patient_dict[i]
            else:
                exclude_indices.append(i)

        celldata = adata.celldata
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        celldata.uns["img_keys"] = list(set(celldata.uns["img_keys"]) - set(exclude_indices))


        print(f"dropped {len(exclude_indices)} images..")
        adata.celldata = celldata
        adata.img_celldata = img_celldata
        adata.img_to_patient_dict = img_to_patient_dict


    # Split data by patients
    if isinstance(test_split, float) and isinstance(val_split, float):
        
        idx_train, idx_test, idx_val = _split_data(
            adata=adata,
            test_split=test_split,
            validation_split=val_split,
            seed_test=seed_test,
            seed_val=seed_val,
        )
    
    adatas = list(adata.img_celldata.values())

    # drop adata with nans in any graph label
    """adatas_filtered = []
    for adata in adatas:
        keep = True
        if isinstance(key_graph_supervision, list):
            for key in key_graph_supervision:
                key = key.split("/")[-1]
                if np.isnan(adata.uns["graph_covariates"]["label_tensors"][key]).any():
                    keep = False
                    break
        else:
            key = key_graph_supervision.split("/")[-1]
            if np.isnan(adata.uns["graph_covariates"]["label_tensors"][key]).any():
                keep = False
                break
        if keep:
            adatas_filtered.append(adata)
    adatas = adatas_filtered.copy()"""




    adatas_ss = []
    if key_local_supervision and key_local_assignment:
        key = key_local_supervision.split("/")[-1]
        key_assignment = key_local_assignment.split("/")[-1]
        for adata in adatas:
            local_supervision, local_assigment = get_self_supervision_label(adata, key, n_clusters=n_clusters)
            adata.uns["graph_covariates"]["label_tensors"][key] = local_supervision
            adata.obsm[key_assignment] = np.argmax(local_assigment, axis=1)
            adatas_ss.append(adata)
        adatas = adatas_ss.copy()

    dm = get_datamodule_from_adata(
        adatas=adatas, 
        key_x=key_x,
        key_graph_supervision=key_graph_supervision,
        batch_size=batch_size,
        edge_index=edge_index,
        key_local_assignment=key_local_assignment,
        key_local_supervision=key_local_supervision,
        key_node_supervision=key_node_supervision,
        num_workers=num_workers,
        preprocess=preprocess,
        val_split=val_split,
        test_split=test_split,
        idx_train=idx_train,
        idx_test=idx_test,
        idx_val=idx_val,
    )
    return dm

def _split_data(
            adata: ad.AnnData,
            test_split: float,
            validation_split: float = 0.0,
            seed_test: int = 1,
            seed_val: int = 2,
    ):
        """
        Split data randomly into partitions.
        :param test_split: Fraction of total data to be in test set.
        :param validation_split: Fraction of train-eval data to be in validation split.
        :param seed: Seed for random selection of observations.
        :return:
        """
        # Do Test-Val-Train split by patients and put all images for a patient into the chosen partition
        np.random.seed(seed_test)

        patient_ids_unique = np.unique(list(adata.img_to_patient_dict.values()))

        number_patients_test = round(len(patient_ids_unique) * test_split)
        patient_keys_test = patient_ids_unique[np.random.choice(
            a=np.arange(len(patient_ids_unique)),
            size=number_patients_test,
            replace=False
        )]

        np.random.seed(seed_val)
        patient_idx_train_eval = np.array([x for x in patient_ids_unique if x not in patient_keys_test])
        number_patients_eval = round(len(patient_idx_train_eval) * validation_split)
        patient_keys_eval = patient_idx_train_eval[np.random.choice(
            a=np.arange(len(patient_idx_train_eval)),
            size=number_patients_eval,
            replace=False
        )]

        patient_keys_train = np.array([x for x in patient_idx_train_eval if x not in patient_keys_eval])

        img_keys_train, img_keys_test, img_keys_val = _split_data_by_patients(
            adata,
            patient_keys_test,
            patient_keys_eval,
            patient_keys_train
        )

        train_idx = []
        test_idx = []
        val_idx = []

        # map img keys to indices
        for key in img_keys_train:
            train_idx.append(list(adata.img_celldata.keys()).index(key))

        for key in img_keys_test:
            test_idx.append(list(adata.img_celldata.keys()).index(key))   
        
        for key in img_keys_val:
            val_idx.append(list(adata.img_celldata.keys()).index(key))    

        return train_idx, test_idx, val_idx

        

def _split_data_by_patients(
        adata: ad.AnnData,
        patient_keys_test,
        patient_keys_eval,
        patient_keys_train,
):
    """
    Split data into partitions defined by user arguments.
    :param patient_idx_test:
    :param patient_idx_val:
    :param patient_idx_train:
    :return:
    """
    patient_to_imagelist = {}
    for patient in np.unique(list(adata.img_to_patient_dict.values())):
        patient_to_imagelist[patient] = []
    for image, patient in adata.img_to_patient_dict.items():
        patient_to_imagelist[patient].append(image)

    adata.img_keys_train = np.concatenate([
        patient_to_imagelist[patient] for patient in patient_keys_train
    ])
    if len(patient_keys_test) > 0:
        adata.img_keys_test = np.concatenate([
            patient_to_imagelist[patient] for patient in patient_keys_test
        ])
        
    else:
        adata.img_keys_test = []
    if len(patient_keys_eval) > 0:
        adata.img_keys_eval = np.concatenate([
            patient_to_imagelist[patient] for patient in patient_keys_eval
        ])
    else:
        adata.img_keys_eval = []

    
    print(
        f"Test dataset: {len(adata.img_keys_test)} images from {len(patient_keys_test)} patients.\n"
        f"Training dataset: {len(adata.img_keys_train)} images from {len(patient_keys_train)} patients.\n"
        f"Validation dataset: {len(adata.img_keys_eval)} images from {len(patient_keys_eval)} patients.\n"
    )
    if len(adata.img_keys_train) == 0:
        raise ValueError("The train dataset is empty.")


    return adata.img_keys_train, adata.img_keys_test, adata.img_keys_eval
