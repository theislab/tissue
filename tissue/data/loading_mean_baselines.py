from typing import List, Union

import anndata as ad
import numpy as np

from tissue_pytorch.data.datasets import DatasetBaselZurichZenodo, DatasetMetabric, DatasetSchuerch
from tissue_pytorch.data.utils import split_data
from tissue_pytorch.data.clustering import get_self_supervision_label

from tissue_pytorch.imports.transforms import get_adjacency_from_adata


class Data:
    def __init__(self,
            adatas: List[ad.AnnData],
            node_feature_space: str, #['molecular', 'celltype']
            key_supervision: str,
            val_split: float = 0.1,
            test_split: float = 0.1,
            dispersion: bool = False,
            dispersion_label: str = "relative_cell_types",
            n_cluster: int = 5,
            node_degree: bool = False,
            percentile: float = 75,
            seed_test: int = 42,
            seed_val: int = 7,
            ):
        
        self.adatas = adatas
        self.node_feature_space = node_feature_space
        self.key_supervision = key_supervision
        self.val_split = val_split
        self.test_split = test_split
        self.dispersion = dispersion
        self.dispersion_label = dispersion_label
        self.n_cluster = n_cluster
        self.node_degree = node_degree
        self.percentile = percentile
        self.seed_test = seed_test
        self.seed_val = seed_val

        self._get_data_from_adata()


        

    def _get_data_from_adata(
            self,
    ):
        
        if self.node_feature_space == "molecular":
            self.X = {k: adata.X.mean(axis=0) for k, adata in self.adatas.img_celldata.items()}
        elif self.node_feature_space == "celltype":
            self.X = {k: adata.obsm["node_types"].mean(axis=0) for k, adata in self.adatas.img_celldata.items()}
        else:
            raise ValueError(f"{self.node_feature_space} not defined")

        self.y = {
                img: imgdata.uns['graph_covariates']['label_tensors'][self.key_supervision]
                for img, imgdata in self.adatas.img_celldata.items()
            }

        # for dispersion models - cell type only
        if self.dispersion:
            self.X = {}
            for k, adata in self.adatas.img_celldata.items():
                rel_labels, _ = get_self_supervision_label(adata, label=self.dispersion_label, n_clusters=self.n_cluster)
                self.X[k] = rel_labels.mean(axis=0)

        # for spatial only models - node degree (node_feature_space is ignored)
        if self.node_degree:
            self.X = {}
            for k, adata in self.adatas.img_celldata.items():
                adj_matrix = get_adjacency_from_adata(adata)
                degrees = np.sum(adj_matrix, axis=1)
        #         print(degrees)
                degree_percentile = np.percentile(np.array(degrees).flatten(), self.percentile)
                self.X[k] = degree_percentile


        # split data by patients
        self.train_img_keys, self.test_img_keys, self.val_img_keys = split_data(self.adatas, 
                                                        test_split=self.test_split, 
                                                        validation_split=self.val_split,
                                                        img_keys=True,
                                                        seed_test=self.seed_test,
                                                        seed_val=self.seed_val
                                                        )

        self.X_train = np.array([])
        self.y_train = np.array([])

        self.X_test = np.array([])
        self.y_test = np.array([])

        self.X_val = np.array([])
        self.y_val = np.array([])


        for key in self.train_img_keys:
            if len(self.X_train) > 0:
                self.X_train = np.vstack((self.X_train, [self.X[key]]))
            else:
               self.X_train = np.array([self.X[key]])
            self.y_train = np.append(self.y_train, self.y[key].argmax())
            
            
        for key in self.test_img_keys:
            if len(self.X_test) > 0:
                self.X_test = np.vstack((self.X_test, [self.X[key]]))
            else:
                self.X_test = np.array([self.X[key]])
            self.y_test = np.append(self.y_test, self.y[key].argmax())
            

        if len(self.val_img_keys) > 0:
            for key in self.val_img_keys:
                if len(self.X_val) > 0:
                    self.X_val = np.vstack((self.X_val, [self.X[key]]))
                else:
                    self.X_val = np.array([self.X[key]])
                self.y_val = np.append(self.y_val, self.y[key].argmax())




def get_data_from_curated(
        dataset: str,
        data_path: str, 
        radius: int,

        node_feature_space: str,
        key_supervision: str,
        
        buffered_data_path: str = None,
        cell_type_coarseness: str = "fine", 

        val_split: float = 0.1,
        test_split: float = 0.1,
        
        dispersion: bool = False,
        dispersion_label: str = "relative_cell_types",
        n_cluster: int = 5,
        node_degree: bool = False,
        percentile: float = 75,
        seed_test: int = 42,
        seed_val: int = 7,
        
) -> Data:
    """
    Args:
    dataset (str): dataset name.
    data_path (str): data path
    radius (int): radius of neighbor range
   
    node_feature_space (str)
    key_supervision (str)
    
    buffered_data_path: str 
    cell_type_coarseness: str = "fine", 

    val_split: float = 0.1,
    test_split: float = 0.1,
    
    dispersion: bool = False,
    n_cluster: int = 5,
    node_degree: bool = False,
    percentile: float = 75,
    seed_test: int = 42,
    seed_val: int = 7
    
    Returns: Instance of Data.
    """
    if dataset.lower() == "schuerch":
        Dataset = DatasetSchuerch
    elif dataset.lower() == "metabric":
        Dataset = DatasetMetabric
    elif dataset.lower() in ["jackson", "baselzurich", "zenodo"]:
        Dataset = DatasetBaselZurichZenodo
    else:
        raise ValueError("Dataset %r is not curated yet." % dataset)
    adatas = Dataset(
        data_path=data_path,
        buffered_data_path=buffered_data_path,
        radius=radius,
        cell_type_coarseness=cell_type_coarseness
    )

    data = Data(
        adatas=adatas,
        node_feature_space=node_feature_space,
        key_supervision=key_supervision,
        val_split=val_split,
        test_split=test_split,
        dispersion=dispersion,
        dispersion_label=dispersion_label,
        n_cluster=n_cluster,
        node_degree=node_degree,
        percentile=percentile,
        seed_test=seed_test,
        seed_val=seed_val
    )
    return data

    


    


