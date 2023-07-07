import os
import abc
import warnings
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import squidpy as sq
from anndata import AnnData, read_h5ad
from pandas import read_csv, read_excel, DataFrame
from scipy import sparse

class GraphTools:
    """GraphTools class."""

    celldata: AnnData
    img_celldata: Dict[str, AnnData]

    def compute_adjacency_matrices(
        self,
        radius: int,
        coord_type: str = "generic",
        n_rings: int = 1,
        transform: str = None,
    ):
        """Compute adjacency matrix for each image in dataset (uses `squidpy.gr.spatial_neighbors`).
        Parameters
        ----------
        radius : int
            Radius of neighbors for non-grid data.
        coord_type : str
            Type of coordinate system.
        n_rings : int
            Number of rings of neighbors for grid data.
        transform : str
            Type of adjacency matrix transform. Valid options are:
            - `spectral` - spectral transformation of the adjacency matrix.
            - `cosine` - cosine transformation of the adjacency matrix.
            - `None` - no transformation of the adjacency matrix.
        """
        for _k, adata in self.img_celldata.items():
            if coord_type == "grid":
                radius = None
            else:
                n_rings = 1
            sq.gr.spatial_neighbors(
                adata=adata,
                coord_type=coord_type,
                radius=radius,
                n_rings=n_rings,
                transform=transform,
                key_added="adjacency_matrix",
            )

    @staticmethod
    def _transform_a(a):
        """Compute degree transformation of adjacency matrix.
        Computes D^(-1) * (A+I), with A an adjacency matrix, I the identity matrix and D the degree matrix.
        Parameters
        ----------
        a
            sparse adjacency matrix.
        Returns
        -------
        degree transformed sparse adjacency matrix
        """
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in true_divide"
        )
        degrees = 1 / a.sum(axis=0)
        degrees[a.sum(axis=0) == 0] = 0
        degrees = np.squeeze(np.asarray(degrees))
        deg_matrix = sparse.diags(degrees)
        a_out = deg_matrix * a
        return a_out

    def _transform_all_a(self, a_dict: dict):
        """Compute degree transformation for dictionary of adjacency matrices.
        Computes D^(-1) * (A+I), with A an adjacency matrix, I the identity matrix and D the degree matrix for all
        matrices in a dictionary.
        Parameters
        ----------
        a_dict : dict
            a_dict
        Returns
        -------
        dictionary of degree transformed sparse adjacency matrices
        """
        a_transformed = {i: self._transform_a(a) for i, a in a_dict.items()}
        return a_transformed

    @staticmethod
    def _compute_distance_matrix(pos_matrix):
        """Compute distance matrix.
        Parameters
        ----------
        pos_matrix
            Position matrix.
        Returns
        -------
        distance matrix
        """
        diff = pos_matrix[:, :, None] - pos_matrix[:, :, None].T
        return (diff * diff).sum(1)

    def _get_degrees(self, max_distances: list):
        """Get dgrees.
        Parameters
        ----------
        max_distances : list
            List of maximal distances.
        Returns
        -------
        degrees
        """
        degs = {}
        degrees = {}
        for i, adata in self.img_celldata.items():
            pm = np.array(adata.obsm["spatial"])
            dist_matrix = self._compute_distance_matrix(pm)
            degs[i] = {
                dist: np.sum(dist_matrix < dist * dist, axis=0)
                for dist in max_distances
            }
        for dist in max_distances:
            degrees[dist] = [deg[dist] for deg in degs.values()]
        return degrees

    def process_node_features(
        self,
        node_feature_transformation: str,
    ):
        # Process node-wise features:
        if node_feature_transformation == "standardize_per_image":
            self._standardize_features_per_image()
        elif node_feature_transformation == "standardize_globally":
            self._standardize_overall()
        elif node_feature_transformation == "scale_observations":
            self._scale_observations()
        elif node_feature_transformation is None or node_feature_transformation == "none":
            pass
        else:
            raise ValueError("Feature transformation %s not recognized!" % node_feature_transformation)

    def _standardize_features_per_image(self):
        for adata in self.img_celldata.values():
            adata.X = adata.X - adata.X.mean(axis=0) / (adata.X.std(axis=0) + 1e-8)

    def _standardize_overall(self):
        data = np.concatenate([adata.X for adata in self.img_celldata.values()], axis=0)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        for adata in self.img_celldata.values():
            adata.X = adata.X - mean / std

    def _scale_observations(self, n: int = 100):
        """
        TPM-like scaling of observation vectors.
        Only makes sense with positive input.
        :param n: Total feature count to linearly scale observations into.
        :return:
        """
        for adata in self.img_celldata.values():
            adata.X = n * adata.X / adata.X.mean(axis=1)


class Dataset(GraphTools):
    """Dataset class. Inherits all functions from GraphTools."""

    def __init__(
        self,
        data_path: str,
        buffered_data_path: str = None,
        write_buffer: bool = False,
        radius: Optional[int] = None,
        coord_type: str = "generic",
        n_rings: int = 1,
        label_selection: Optional[List[str]] = None,
        n_top_genes: Optional[int] = None,
        cell_type_coarseness: str = "fine",
    ):
        """Initialize Dataset.
        Parameters
        ----------
        data_path : str
            Data path.
        radius : int
            Radius.
        label_selection : list, optional
            label selection.
        """
        self.data_path = data_path
        self.buffered_data_path = buffered_data_path
        self.cell_type_coarseness = cell_type_coarseness

        fn = f"{buffered_data_path}/buffered_data_{str(radius)}_{cell_type_coarseness}.pickle"
        if os.path.isfile(fn) and not write_buffer:
            print("Loading data from buffer")
            with open(fn, "rb") as f:
                stored_data = pickle.load(f)
            self.celldata = stored_data["celldata"]
            self.img_celldata = stored_data["img_celldata"]
            self.radius = radius
            self.img_to_patient_dict = stored_data["celldata"].uns["img_to_patient_dict"]

        else:            
            print("Loading data from raw files")
            self.register_celldata(n_top_genes=n_top_genes)
            self.register_img_celldata()
            self.register_graph_features(label_selection=label_selection)
            self.compute_adjacency_matrices(radius=radius, coord_type=coord_type, n_rings=n_rings)
            self.radius = radius
            
            if write_buffer: 
                print("Buffering preprocessed input data")
                data_to_store = {
                    "celldata":self.celldata,
                    "img_celldata": self.img_celldata,
                    "img_to_patient_dict": self.img_to_patient_dict,
                }
                with open(fn, "wb") as f:
                    pickle.dump(obj=data_to_store, file=f)

        print(
            "Loaded %i images with complete data from %i patients "
            "over %i cells with %i cell features and %i distinct celltypes."
            % (
                len(self.img_celldata),
                len(self.patients),
                self.celldata.shape[0],
                self.celldata.shape[1],
                len(self.celldata.uns["node_type_names"]),
            )
        )

    @property
    def patients(self):
        """Return number of patients in celldata.
        Returns
        -------
        patients
        """
        return np.unique(
            np.asarray(list(self.celldata.uns["img_to_patient_dict"].values()))
        )

    def register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        print("registering celldata")
        self._register_celldata(n_top_genes=n_top_genes)
        assert self.celldata is not None, "celldata was not loaded"

    def register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        print("collecting image-wise celldata")
        self._register_img_celldata()
        assert self.img_celldata is not None, "image-wise celldata was not loaded"

    def register_graph_features(self, label_selection):
        """Load graph level covariates.
        Parameters
        ----------
        label_selection
            Label selection.
        """
        print("adding graph-level covariates")
        self._register_graph_features(label_selection=label_selection)

    @abc.abstractmethod
    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        pass

    @abc.abstractmethod
    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        pass

    @abc.abstractmethod
    def _register_graph_features(self, label_selection):
        """Load graph level covariates.
        Parameters
        ----------
        label_selection
            Label selection.
        """
        pass

    def size_factors(self):
        """Get size factors (Only makes sense with positive input).
        Returns
        -------
        sf_dict
        """
        # Check if irregular sums are encountered:
        for i, adata in self.img_celldata.items():
            if np.any(np.sum(adata.X, axis=1) <= 0):
                print("WARNING: found irregular node sizes in image %s" % str(i))
        # Get global mean of feature intensity across all features:
        global_mean_per_node = self.celldata.X.sum(axis=1).mean(axis=0)
        return {
            i: global_mean_per_node / np.sum(adata.X, axis=1)
            for i, adata in self.img_celldata.items()
        }

    @property
    def var_names(self):
        return self.celldata.var_names



class DatasetSchuerch(Dataset):
    """DatasetSchuerch class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "fine": {
            "B cells": "B cells",
            "CD11b+ monocytes": "monocytes",
            "CD11b+CD68+ macrophages": "macrophages",
            "CD11c+ DCs": "dendritic cells",
            "CD163+ macrophages": "macrophages",
            "CD3+ T cells": "CD3+ T cells",
            "CD4+ T cells": "CD4+ T cells",
            "CD4+ T cells CD45RO+": "CD4+ T cells",
            "CD4+ T cells GATA3+": "CD4+ T cells",
            "CD68+ macrophages": "macrophages",
            "CD68+ macrophages GzmB+": "macrophages",
            "CD68+CD163+ macrophages": "macrophages",
            "CD8+ T cells": "CD8+ T cells",
            "NK cells": "NK cells",
            "Tregs": "Tregs",
            "adipocytes": "adipocytes",
            "dirt": "dirt",
            "granulocytes": "granulocytes",
            "immune cells": "immune cells",
            "immune cells / vasculature": "immune cells",
            "lymphatics": "lymphatics",
            "nerves": "nerves",
            "plasma cells": "plasma cells",
            "smooth muscle": "smooth muscle",
            "stroma": "stroma",
            "tumor cells": "tumor cells",
            "tumor cells / immune cells": "immune cells",
            "undefined": "undefined",
            "vasculature": "vasculature",
        },
        "binary": {
            "B cells": "immune cells",
            "CD11b+ monocytes": "immune cells",
            "CD11b+CD68+ macrophages": "immune cells",
            "CD11c+ DCs": "immune cells",
            "CD163+ macrophages": "immune cells",
            "CD3+ T cells": "immune cells",
            "CD4+ T cells": "immune cells",
            "CD4+ T cells CD45RO+": "immune cells",
            "CD4+ T cells GATA3+": "immune cells",
            "CD68+ macrophages": "immune cells",
            "CD68+ macrophages GzmB+": "immune cells",
            "CD68+CD163+ macrophages": "immune cells",
            "CD8+ T cells": "immune cells",
            "NK cells": "immune cells",
            "Tregs": "immune cells",
            "adipocytes": "other",
            "dirt": "other",
            "granulocytes": "immune cells",
            "immune cells": "immune cells",
            "immune cells / vasculature": "immune cells",
            "lymphatics": "immune cells",
            "nerves": "other",
            "plasma cells": "other",
            "smooth muscle": "other",
            "stroma": "other",
            "tumor cells": "other",
            "tumor cells / immune cells": "immune cells",
            "undefined": "other",
            "vasculature": "other"
        },
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.377442,
            "fn": "CRC_clusters_neighborhoods_markers.csv", # previously "fn": "CRC_clusters_neighborhoods_markers_NEW.csv"
            "image_col": "File Name",
            "pos_cols": ["X:X", "Y:Y"],
            "cluster_col": "ClusterName",
            "cluster_col_preprocessed": "ClusterName_preprocessed",
            "patient_col": "patients",
            "cell_type_coarseness": self.cell_type_coarseness,
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        feature_cols = [
            "CD44 - stroma:Cyc_2_ch_2",
            "FOXP3 - regulatory T cells:Cyc_2_ch_3",
            "CD8 - cytotoxic T cells:Cyc_3_ch_2",
            "p53 - tumor suppressor:Cyc_3_ch_3",
            "GATA3 - Th2 helper T cells:Cyc_3_ch_4",
            "CD45 - hematopoietic cells:Cyc_4_ch_2",
            "T-bet - Th1 cells:Cyc_4_ch_3",
            "beta-catenin - Wnt signaling:Cyc_4_ch_4",
            "HLA-DR - MHC-II:Cyc_5_ch_2",
            "PD-L1 - checkpoint:Cyc_5_ch_3",
            "Ki67 - proliferation:Cyc_5_ch_4",
            "CD45RA - naive T cells:Cyc_6_ch_2",
            "CD4 - T helper cells:Cyc_6_ch_3",
            "CD21 - DCs:Cyc_6_ch_4",
            "MUC-1 - epithelia:Cyc_7_ch_2",
            "CD30 - costimulator:Cyc_7_ch_3",
            "CD2 - T cells:Cyc_7_ch_4",
            "Vimentin - cytoplasm:Cyc_8_ch_2",
            "CD20 - B cells:Cyc_8_ch_3",
            "LAG-3 - checkpoint:Cyc_8_ch_4",
            "Na-K-ATPase - membranes:Cyc_9_ch_2",
            "CD5 - T cells:Cyc_9_ch_3",
            "IDO-1 - metabolism:Cyc_9_ch_4",
            "Cytokeratin - epithelia:Cyc_10_ch_2",
            "CD11b - macrophages:Cyc_10_ch_3",
            "CD56 - NK cells:Cyc_10_ch_4",
            "aSMA - smooth muscle:Cyc_11_ch_2",
            "BCL-2 - apoptosis:Cyc_11_ch_3",
            "CD25 - IL-2 Ra:Cyc_11_ch_4",
            "CD11c - DCs:Cyc_12_ch_3",
            "PD-1 - checkpoint:Cyc_12_ch_4",
            "Granzyme B - cytotoxicity:Cyc_13_ch_2",
            "EGFR - signaling:Cyc_13_ch_3",
            "VISTA - costimulator:Cyc_13_ch_4",
            "CD15 - granulocytes:Cyc_14_ch_2",
            "ICOS - costimulator:Cyc_14_ch_4",
            "Synaptophysin - neuroendocrine:Cyc_15_ch_3",
            "GFAP - nerves:Cyc_16_ch_2",
            "CD7 - T cells:Cyc_16_ch_3",
            "CD3 - T cells:Cyc_16_ch_4",
            "Chromogranin A - neuroendocrine:Cyc_17_ch_2",
            "CD163 - macrophages:Cyc_17_ch_3",
            "CD45RO - memory cells:Cyc_18_ch_3",
            "CD68 - macrophages:Cyc_18_ch_4",
            "CD31 - vasculature:Cyc_19_ch_3",
            "Podoplanin - lymphatics:Cyc_19_ch_4",
            "CD34 - vasculature:Cyc_20_ch_3",
            "CD38 - multifunctional:Cyc_20_ch_4",
            "CD138 - plasma cells:Cyc_21_ch_3",
            "HOECHST1:Cyc_1_ch_1",
            "CDX2 - intestinal epithelia:Cyc_2_ch_4",
            "Collagen IV - bas. memb.:Cyc_12_ch_2",
            "CD194 - CCR4 chemokine R:Cyc_14_ch_3",
            "MMP9 - matrix metalloproteinase:Cyc_15_ch_2",
            "CD71 - transferrin R:Cyc_15_ch_4",
            "CD57 - NK cells:Cyc_17_ch_4",
            "MMP12 - matrix metalloproteinase:Cyc_21_ch_4",
        ]
        feature_cols_hgnc_names = [
            "CD44",
            "FOXP3",
            "CD8A",
            "TP53",
            "GATA3",
            "PTPRC",
            "TBX21",
            "CTNNB1",
            "HLA-DR",
            "CD274",
            "MKI67",
            "PTPRC",
            "CD4",
            "CR2",
            "MUC1",
            "TNFRSF8",
            "CD2",
            "VIM",
            "MS4A1",
            "LAG3",
            "ATP1A1",
            "CD5",
            "IDO1",
            "KRT1",
            "ITGAM",
            "NCAM1",
            "ACTA1",
            "BCL2",
            "IL2RA",
            "ITGAX",
            "PDCD1",
            "GZMB",
            "EGFR",
            "VISTA",
            "FUT4",
            "ICOS",
            "SYP",
            "GFAP",
            "CD7",
            "CD247",
            "CHGA",
            "CD163",
            "PTPRC",
            "CD68",
            "PECAM1",
            "PDPN",
            "CD34",
            "CD38",
            "SDC1",
            "HOECHST1:Cyc_1_ch_1",  ##
            "CDX2",
            "COL6A1",
            "CCR4",
            "MMP9",
            "TFRC",
            "B3GAT1",
            "MMP12"
        ]
        X = DataFrame(np.array(celldata_df[feature_cols]), columns=feature_cols_hgnc_names)
        celldata = AnnData(X=X, obs=celldata_df[["File Name", "patients", "ClusterName"]])
        celldata.var_names_make_unique()

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "category"
        )

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.
        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Graph features are based on TMA spot and not patient, thus patient_col is technically wrong.
        # For aspects where patients are needed (e.g. train-val-test split) the correct patients that are
        # loaded in _register_images() are used
        patient_col = "TMA spot / region"
        disease_features = {}
        patient_features = {"Sex": "categorical", "Age": "continuous"}
        survival_features = {"DFS": "survival"}
        tumor_features = {
            # not sure where these features belong
            "Group": "categorical",
            "LA": "percentage",
            "Diffuse": "percentage",
            "Klintrup_Makinen": "categorical",
            "CLR_Graham_Appelman": "categorical",
        }
        treatment_features = {}
        col_renaming = {}

        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)
        label_cols.update(survival_features)
        label_cols.update(tumor_features)
        label_cols.update(treatment_features)

        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(label_selection.intersection(set(list(label_cols.keys()))))
        if "DFS" in label_selection:
            censor_col = "DFS_Censor"
            label_cols_toread = label_cols_toread + [censor_col]
        # there are two LA and Diffuse columns for the two cores that are represented by one patient row
        if "LA" in label_cols_toread:
            label_cols_toread = label_cols_toread + ["LA.1"]
        if "Diffuse" in label_cols_toread:
            label_cols_toread = label_cols_toread + ["Diffuse.1"]
        label_cols_toread_csv = [
            col_renaming[col] if col in list(col_renaming.keys()) else col for col in label_cols_toread
        ]

        usecols = label_cols_toread_csv + [patient_col]
        tissue_meta_data = read_csv(
            os.path.join(self.data_path, "CRC_TMAs_patient_annotations.csv"),
            # sep="\t",
            usecols=usecols,
        )[usecols]
        tissue_meta_data.columns = label_cols_toread + [patient_col]

        # preprocess the loaded csv data:
        # the rows after the first 35 are just descriptions that were included in the excel file
        # for easier work with the data, we expand the data to have two columns per patient representing the two cores
        # that have different LA and Diffuse labels
        patient_data = tissue_meta_data[:35]
        long_patient_data = pd.DataFrame(np.repeat(patient_data.values, 2, axis=0))
        long_patient_data.columns = patient_data.columns
        long_patient_data["copy"] = ["A", "B"] * 35
        if "Diffuse" in label_cols_toread:
            long_patient_data = long_patient_data.rename(columns={"Diffuse": "DiffuseA", "Diffuse.1": "DiffuseB"})
            long_patient_data["Diffuse"] = np.zeros((70,))
            long_patient_data.loc[long_patient_data["copy"] == "A", "Diffuse"] = long_patient_data[
                long_patient_data["copy"] == "A"
            ]["DiffuseA"]
            long_patient_data.loc[long_patient_data["copy"] == "B", "Diffuse"] = long_patient_data[
                long_patient_data["copy"] == "B"
            ]["DiffuseB"]
            long_patient_data.loc[long_patient_data["Diffuse"].isnull(), "Diffuse"] = 0
            # use the proportion of diffuse cores within this spot as probability of being diffuse
            long_patient_data["Diffuse"] = long_patient_data["Diffuse"].astype(float) / 2
            long_patient_data = long_patient_data.drop("DiffuseA", axis=1)
            long_patient_data = long_patient_data.drop("DiffuseB", axis=1)
        if "LA" in label_cols_toread:
            long_patient_data = long_patient_data.rename(columns={"LA": "LAA", "LA.1": "LAB"})
            long_patient_data["LA"] = np.zeros((70,))
            long_patient_data.loc[long_patient_data["copy"] == "A", "LA"] = long_patient_data[
                long_patient_data["copy"] == "A"
            ]["LAA"]
            long_patient_data.loc[long_patient_data["copy"] == "B", "LA"] = long_patient_data[
                long_patient_data["copy"] == "B"
            ]["LAB"]
            long_patient_data.loc[long_patient_data["LA"].isnull(), "LA"] = 0
            # use the proportion of LA cores within this spot as probability of being LA
            long_patient_data["LA"] = long_patient_data["LA"].astype(float) / 2
            long_patient_data = long_patient_data.drop("LAA", axis=1)
            long_patient_data = long_patient_data.drop("LAB", axis=1)
        tissue_meta_data = long_patient_data

        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {label: type for label, type in label_cols.items() if label in label_selection}
        label_tensors = {}
        label_names = {}  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "continuous":
                label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) / continuous_std[
                    feature
                ]
                label_names[feature] = [feature]
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "percentage":
                label_tensors[feature] = tissue_meta_data[feature]
                label_names[feature] = [feature]
        # 2. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "categorical":
                tissue_meta_data[feature] = tissue_meta_data[feature].astype("str")
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "categorical":
                oh = pd.get_dummies(tissue_meta_data[feature], prefix=feature, prefix_sep=">", drop_first=False, dtype=float)
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array([i for i, x in enumerate(oh.columns) if x.endswith(">nan")])
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.0)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith(">nan")]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # 3. Add censoring information to survival
        # survival_mean = {
        #     feature: tissue_meta_data[feature].mean(skipna=True)
        #     for feature in list(label_cols.keys())
        #     if label_cols[feature] == "survival"
        # }
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "survival":
                label_tensors[feature] = np.concatenate(
                    [
                        np.expand_dims(tissue_meta_data[feature].values, axis=1),
                        np.expand_dims(tissue_meta_data[censor_col].values, axis=1),
                    ],
                    axis=1,
                )
                label_names[feature] = [feature]
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        # tissue_meta_data_patients = tissue_meta_data[patient_col].values.tolist()
        # image keys are of the form reg0xx_A or reg0xx_B with xx going from 01 to 70
        # label tensors have entries (1+2)_A, (1+2)_B, (2+3)_A, (2+3)_B, ...
        img_to_index = {
            img: 2 * ((int(img[4:6]) - 1) // 2) if img[7] == "A" else 2 * ((int(img[4:6]) - 1) // 2) + 1
            for img in self.img_to_patient_dict.keys()
        }
        label_tensors = {
            img: {
                feature_name: np.array(features[index, :], ndmin=1) for feature_name, features in label_tensors.items()
            }
            for img, index in img_to_index.items()
        }

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DatasetBaselZurichZenodo(Dataset):
    """DatasetBaselZurichZenodo class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "fine": {
            "1": "B cells",
            "2": "T and B cells",
            "3": "T cells",
            "4": "macrophages",
            "5": "T cells",
            "6": "macrophages",
            "7": "endothelial",
            "8": "stromal cells", # "vimentin hi stromal cell",
            "9": "stromal cells", # "small circular stromal cell",
            "10": "stromal cells", # "small elongated stromal cell",
            "11": "stromal cells", # "fibronectin hi stromal cell",
            "12": "stromal cells", # "large elongated stromal cell",
            "13": "stromal cells", # "SMA hi vimentin hi stromal cell",
            "14": "tumor cells", #"hypoxic tumor cell",
            "15": "tumor cells", #"apoptotic tumor cell",
            "16": "tumor cells", #"proliferative tumor cell",
            "17": "tumor cells", #"p53+ EGFR+ tumor cell",
            "18": "tumor cells", #"basal CK tumor cell",
            "19": "tumor cells", #"CK7+ CK hi cadherin hi tumor cell",
            "20": "tumor cells", #"CK7+ CK+ tumor cell",
            "21": "tumor cells", #"epithelial low tumor cell",
            "22": "tumor cells", #"CK low HR low tumor cell",
            "23": "tumor cells", #"CK+ HR hi tumor cell",
            "24": "tumor cells", #"CK+ HR+ tumor cell",
            "25": "tumor cells", #"CK+ HR low tumor cell",
            "26": "tumor cells", #"CK low HR hi p53+ tumor cell",
            "27": "tumor cells", #"myoepithelial tumor cell"
        },
        "binary": {
            "1": "immune cells",
            "2": "immune cells",
            "3": "immune cells",
            "4": "immune cells",
            "5": "immune cells",
            "6": "immune cells",
            "7": "other",
            "8": "other", # "vimentin hi stromal cell",
            "9": "other", # "small circular stromal cell",
            "10": "other", # "small elongated stromal cell",
            "11": "other", # "fibronectin hi stromal cell",
            "12": "other", # "large elongated stromal cell",
            "13": "other", # "SMA hi vimentin hi stromal cell",
            "14": "other", #"hypoxic tumor cell",
            "15": "other", #"apoptotic tumor cell",
            "16": "other", #"proliferative tumor cell",
            "17": "other", #"p53+ EGFR+ tumor cell",
            "18": "other", #"basal CK tumor cell",
            "19": "other", #"CK7+ CK hi cadherin hi tumor cell",
            "20": "other", #"CK7+ CK+ tumor cell",
            "21": "other", #"epithelial low tumor cell",
            "22": "other", #"CK low HR low tumor cell",
            "23": "other", #"CK+ HR hi tumor cell",
            "24": "other", #"CK+ HR+ tumor cell",
            "25": "other", #"CK+ HR low tumor cell",
            "26": "other", #"CK low HR hi p53+ tumor cell",
            "27": "other", #"myoepithelial tumor cell"
        },
    }

    def _register_images(self):
        """
        Creates mapping of full image names to shorter identifiers.
        """

        # Define mapping of image identifiers to numeric identifiers:
        img_tab_basel = read_csv(
            self.data_path + "Data_publication/BaselTMA/Basel_PatientMetadata.csv",
            usecols=["core", "FileName_FullStack", "PID", "diseasestatus"],
            dtype={"core": str, "FileName_FullStack": str, "PID": str, "diseasestatus": str}
        )
        img_tab_basel["PID"] = ["b" + str(p) for p in img_tab_basel["PID"].values]
        img_tab_zurich = read_csv(
            self.data_path + "Data_publication/ZurichTMA/Zuri_PatientMetadata.csv",
            usecols=["core", "FileName_FullStack", "grade", "PID", "location"],
            dtype={"core": str, "FileName_FullStack": str, "grade": str, "PID": str, "location": str}
        )
        img_tab_zurich["PID"] = ["z" + str(p) for p in img_tab_zurich["PID"].values]
        img_tab_zurich["diseasestatus"] = [
            "tumor" if a else "non-tumor" for a in img_tab_zurich["location"] != "[]"
        ]
        img_tab_zurich = img_tab_zurich.drop("location", axis=1)
        # drop Metastasis images
        img_tab_zurich = img_tab_zurich[img_tab_zurich["grade"] != "METASTASIS"].drop("grade", axis=1)
        img_tab_bz = pd.concat([img_tab_basel, img_tab_zurich], axis=0, sort=True, ignore_index=True)
        img_tab_bz = img_tab_bz[img_tab_bz["diseasestatus"] == "tumor"]

        self.img_key_to_fn = dict(img_tab_bz[["core", "FileName_FullStack"]].values)
        self.img_to_patient_dict = dict(img_tab_bz[["core", "PID"]].values)

    def _load_node_positions(self):
        from PIL import Image

        position_matrix = []
        for k, fn in self.img_key_to_fn.items():
            fn = self.data_path + "OMEnMasks/Basel_Zuri_masks/" + fn
            # Mask file have slightly different file name, extended either by _mask or _maks:
            if os.path.exists(".".join(fn.split(".")[:-1]) + "_maks.tiff"):
                fn = ".".join(fn.split(".")[:-1]) + "_maks.tiff"
            elif os.path.exists(".".join(fn.split(".")[:-1]) + "_mask.tiff"):
                fn = ".".join(fn.split(".")[:-1]) + "_mask.tiff"
            else:
                raise ValueError("file %s not found" % fn)

            # Load image from tiff:
            img_array = np.array(Image.open(fn))
            # Throughout all files, nodes are refered to via the string core_id+"_"+str(i) where i is the integer
            # encoding the object in the segmentation mask.
            node_ids_img = np.sort(np.unique(img_array))
            # 0 encodes background:
            node_ids_img = node_ids_img[node_ids_img != 0]
            # Only ranks of objects encoded in masks are used!  # TODO check
            node_ids_rank = np.arange(1, len(node_ids_img) + 1)
            # Drop images with fewer than 100 nodes
            if len(node_ids_rank) < 100:
                continue
            # Find centre of object mask of each node:  # TODO check, rank used
            center_array = [np.where(img_array == node_ids_img[i - 1]) for i in node_ids_rank]
            pm = np.array([[f"{k}_{i+1}", x[0].mean(), x[1].mean()] for i, x in enumerate(center_array)])
            position_matrix.append(pm)
        position_matrix = np.concatenate(position_matrix, axis=0)
        position_matrix = pd.DataFrame(position_matrix, columns=["id", "x", "y"])
        return position_matrix

    def _load_node_features(self):
        full_cell_key_col = "id"  # column with full cell identifier (including image identifier)
        feature_col = "channel"
        signal_col = "mc_counts"

        features_basel = read_csv(
            self.data_path + "Data_publication/BaselTMA/SC_dat.csv",
            usecols=[full_cell_key_col, feature_col, signal_col],
            dtype={full_cell_key_col: str, feature_col: str, signal_col: float}
        )
        features_zurich = read_csv(
            self.data_path + "Data_publication/ZurichTMA/SC_dat.csv",
            usecols=[full_cell_key_col, feature_col, signal_col],
            dtype={full_cell_key_col: str, feature_col: str, signal_col: float}
        )
        features_zb = pd.concat([
            features_basel,
            features_zurich
        ], axis=0, ignore_index=True)
        node_features = features_zb.pivot_table(index="id", columns="channel", values="mc_counts")

        return node_features

    def _load_node_types(self):
        """
        Loads the cell types.
        """
        # Direct meta cluster annotation from main text Fig 1b
        # Also hard coded maps from cluster numbers to annotation as in https://github.com/BodenmillerGroup/SCPathology_publication/blob/4e99e10c2bc6d0f1dd168d534df39870d1ecb549/R/BaselTMA_pipeline.Rmd#L471

        full_cell_key_col = "id"  # column with full cell identifier (including image identifier)
        cluster_col = "cluster"
        node_cluster_basel = read_csv(
            self.data_path + "Cluster_labels/Basel_metaclusters.csv",
            usecols=[full_cell_key_col, cluster_col],
            dtype={full_cell_key_col: str, cluster_col: str}
        )
        node_cluster_zurich = read_csv(
            self.data_path + "Cluster_labels/Zurich_matched_metaclusters.csv",
            usecols=[full_cell_key_col, cluster_col],
            dtype={full_cell_key_col: str, cluster_col: str}
        )
        node_cluster = pd.concat([
            node_cluster_basel,
            node_cluster_zurich
        ], axis=0, ignore_index=True)

        return node_cluster

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        self._register_images()
        position_matrix = self._load_node_positions()
        node_features = self._load_node_features()
        node_types = self._load_node_types()

        node_types.set_index("id", inplace=True)
        position_matrix.set_index("id", inplace=True)
        celldata_df = pd.concat([position_matrix, node_features, node_types], axis=1, ignore_index=False, join="outer")
        celldata_df = celldata_df[celldata_df["x"] == celldata_df["x"]]

        celldata_df["core"] = ["_".join(a.split("_")[:-1]) for a in celldata_df.index]
        celldata_df["PID"] = [self.img_to_patient_dict[c] for c in celldata_df["core"]]

        metadata = {
            "lateral_resolution": None,
            "fn": None,
            "image_col": "core",
            "pos_cols": ["x", "y"],
            "cluster_col": "cluster",
            "cluster_col_preprocessed": "cluster_preprocessed",
            "patient_col": "PID",
            "cell_type_coarseness": self.cell_type_coarseness,
        }

        feature_cols = [
            "1021522Tm169Di EGFR",
            "1031747Er167Di ECadhe",
            "112475Gd156Di Estroge",
            "117792Dy163Di GATA3",
            "1261726In113Di Histone",
            "1441101Er168Di Ki67",
            "174864Nd148Di SMA",
            "1921755Sm149Di Vimenti",
            "198883Yb176Di cleaved",
            "201487Eu151Di cerbB",
            "207736Tb159Di p53",
            "234832Lu175Di panCyto",
            "3111576Nd143Di Cytoker",
            "Nd145Di Twist",
            "312878Gd158Di Progest",
            "322787Nd150Di cMyc",
            "3281668Nd142Di Fibrone",
            "346876Sm147Di Keratin",
            "3521227Gd155Di Slug",
            "361077Dy164Di CD20",
            "378871Yb172Di vWF",
            "473968La139Di Histone",
            "651779Pr141Di Cytoker",
            "6967Gd160Di CD44",
            "71790Dy162Di CD45",
            "77877Nd146Di CD68",
            "8001752Sm152Di CD3epsi",
            "92964Er166Di Carboni",
            "971099Nd144Di Cytoker",
            "98922Yb174Di Cytoker",
            "phospho Histone",
            "phospho S6",
            "phospho mTOR",
            "Area"
        ]

        X = DataFrame(np.array(celldata_df[feature_cols]), columns=feature_cols)
        celldata = AnnData(X=X, obs=celldata_df[[metadata["image_col"], metadata["patient_col"], metadata["cluster_col"]]])
        celldata.var_names_make_unique()

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]].astype(
            "category"
        )

        # register node type names
        types = celldata.obs[metadata["cluster_col_preprocessed"]]

        node_type_names = list(np.unique(types[types == types]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) if x == x else 0
                for x in types
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = types == types
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.
        Parameters
        ----------
        label_selection
            Label selection.
        """
        # DEFINE COLUMN NAMES FOR TABULAR DATA.
        # Define column names to extract from patient-wise tabular data:
        patient_col = "PID"
        img_key_col = "core"
        # These are required to assign the image to dieased and non-diseased:
        image_status_cols = ["location", "diseasestatus"]
        # Labels are defined as a column name and a label type:
        disease_features = {
            "grade": "categorical",
            "grade_collapsed": "categorical",
            "tumor_size": "continuous",
            "diseasestatus": "categorical",
            "location": "categorical",
            "tumor_type": "categorical"
        }
        patient_features = {
            "age": "continuous"
        }
        survival_features = {
            "Patientstatus": "categorical",
            "DFSmonth": "survival",
            "OSmonth": "survival"
        }
        tumor_featues = {
            "clinical_type": "categorical",
            "Subtype": "categorical",
            "PTNM_M": "categorical",
            "PTNM_T": "categorical",
            "PTNM_N": "categorical",
            "PTNM_Radicality": "categorical",
            "Lymphaticinvasion": "categorical",
            "Venousinvasion": "categorical",
            "ERStatus": "categorical",
            "PRStatus": "categorical",
            "HER2Status": "categorical",
            # "ER+DuctalCa": "categorical",
            "TripleNegDuctal": "categorical",
            # "hormonesensitive": "categorical",
            # "hormoneresistantaftersenstive": "categorical",
            "microinvasion": "categorical",
            "I_plus_neg": "categorical",
            "SN": "categorical",
            # "MIC": "categorical"
        }
        treatment_feature = {
            "Pre-surgeryTx": "categorical",
            "Post-surgeryTx": "categorical"
        }
        batch_features = {
            "TMABlocklabel": "categorical",
            "Yearofsamplecollection": "continuous"
        }
        ncell_features = {  # not used right now
            "%tumorcells": "percentage",
            "%normalepithelialcells": "percentage",
            "%stroma": "percentage",
            "%inflammatorycells": "percentage",
            "Count_Cells": "continuous"
        }
        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)
        label_cols.update(survival_features)
        label_cols.update(tumor_featues)
        label_cols.update(treatment_feature)
        label_cols.update(batch_features)
        label_cols.update(ncell_features)
        # Clean selected labels based on defined labels:
        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(label_selection.intersection(set(list(label_cols.keys()))))
        # Make sure censoring information is read if surival is predicted:
        if "DFSmonth" in label_cols_toread:
            if "OSmonth" not in label_cols_toread:
                label_cols_toread.append("OSmonth")
        if "OSmonth" in label_cols_toread:
            if "Patientstatus" not in label_cols_toread:
                label_cols_toread.append("Patientstatus")
        if "grade_collapsed" in label_selection and "grade" not in label_selection:
            label_cols_toread.append("grade")

        # READ RAW LABELS, COVARIATES AND IDENTIFIERS FROM TABLES.
        # Read all labels and image- and patient-identifiers from table. This full set is overlapped to the existing
        # columns of file so that files with different column spaces can be read.
        # The patients are renamed for each patient set with a prefix to guarantee uniqueness.
        # The output of this workflow is (1) a single table with rows for each image and with all columns modified
        # so that the can further be processed to tensors of labels and covariates through GLM formula-like commands and
        # (2) indices of diseased and non-diseased images in this table.
        cols_toread = [patient_col, img_key_col] + image_status_cols + label_cols_toread  # full list of columns to read
        # Read Basel data.
        cols_found_basel = read_csv(self.data_path + "Data_publication/BaselTMA/Basel_PatientMetadata.csv", nrows=0)
        cols_toread_basel = set(cols_found_basel.columns) & set(cols_toread)
        tissue_meta_data_basel = read_csv(
            self.data_path + "Data_publication/BaselTMA/Basel_PatientMetadata.csv",
            usecols=cols_toread_basel
        )
        tissue_meta_data_basel[patient_col] = ["b" + str(x) for x in tissue_meta_data_basel[patient_col].values]
        # Read Zuri data.
        cols_found_zuri = read_csv(self.data_path + "Data_publication/ZurichTMA/Zuri_PatientMetadata.csv", nrows=0)
        cols_toread_zuri = set(cols_found_zuri.columns) & set(cols_toread)
        tissue_meta_data_zuri = read_csv(
            self.data_path + "Data_publication/ZurichTMA/Zuri_PatientMetadata.csv",
            usecols=cols_toread_zuri
        )
        tissue_meta_data_zuri[patient_col] = ["z" + str(x) for x in tissue_meta_data_zuri[patient_col].values]

        # Modify specific columns:
        # The diseasestatus is not given in the Zuri data but can be inferred from the location column.
        tissue_meta_data_zuri["diseasestatus"] = [
            "tumor" if a else "non-tumor" for a in tissue_meta_data_zuri["location"] != "[]"
        ]
        # Tumor size is masked if the image does not contain a tumor:
        if "tumor_size" in label_selection:
            no_tumor = list(tissue_meta_data_basel["diseasestatus"] == "non-tumor")
            tissue_meta_data_basel.loc[no_tumor, "tumor_size"] = np.nan
        # Add missing Patientstatus and survival labels in Zuri data that are only given in Basel data set:
        if "Patientstatus" in label_selection:
            tissue_meta_data_zuri["Patientstatus"] = np.nan
        if "OSmonth" in label_selection:
            tissue_meta_data_zuri["OSmonth"] = np.nan
        if "DFSmonth" in label_selection:
            tissue_meta_data_zuri["DFSmonth"] = np.nan
        # Add censoring column if survival is given:
        # All states recorded: alive, alive w metastases, death, death by primary disease
        # Also densor non-disease caused death.
        if "OSmonth" in label_selection:
            tissue_meta_data_basel["censor_OS"] = [
                0 if x in ["alive", "alive w metastases"] else 1  # penalty-scale for over-estimation
                for x in tissue_meta_data_basel["Patientstatus"].values
            ]
            tissue_meta_data_zuri["censor_OS"] = np.nan

        if "DFSmonth" in label_selection:
            tissue_meta_data_basel["censor_DFS"] = [
                0 if tissue_meta_data_basel["OSmonth"][idx] == tissue_meta_data_basel["DFSmonth"][idx] else 1
                for idx in tissue_meta_data_basel["OSmonth"].index
            ]
            tissue_meta_data_zuri["censor_DFS"] = np.nan

        # Replace missing observations labeled as "[]" for PTNM_N, PTNM_M, PTNM_T
        if "PTNM_N" in label_selection:
            tissue_meta_data_zuri["PTNM_N"] = [a[1:] for a in tissue_meta_data_zuri["PTNM_N"]]
            tissue_meta_data_zuri["PTNM_N"].replace("]", "nan", inplace=True)
        if "PTNM_M" in label_selection:
            tissue_meta_data_zuri["PTNM_M"] = [a[1:] for a in tissue_meta_data_zuri["PTNM_M"]]
            tissue_meta_data_zuri["PTNM_M"].replace("]", "nan", inplace=True)
        if "PTNM_T" in label_selection:
            tissue_meta_data_zuri["PTNM_T"] = [a[1:] for a in tissue_meta_data_zuri["PTNM_T"]]
            tissue_meta_data_zuri["PTNM_T"].replace("]", "nan", inplace=True)

        # Merge Basel and Zuri data.
        tissue_meta_data = pd.concat([
            tissue_meta_data_basel,
            tissue_meta_data_zuri
        ], axis=0, sort=True, ignore_index=True)

        if "grade_collapsed" in label_selection:
            tissue_meta_data["grade_collapsed"] = ["3" if grade == "3" else "1&2" for grade in
                                                   tissue_meta_data["grade"]]

        # Drop already excluded images (e.g. METASTASIS or to few nodes)
        tissue_meta_data = tissue_meta_data[tissue_meta_data[img_key_col].isin(list(self.img_to_patient_dict.keys()))].reset_index()

        # Final processing:
        # Remove columns that are only used to infer missing entries in other columns:
        if "location" not in label_selection:
            tissue_meta_data.drop("location", 1, inplace=True)
        if "grade_collapsed" in label_selection and "grade" not in label_selection:
            tissue_meta_data.drop("grade", 1, inplace=True)
        # Some non-label columns remain in the table as these are used to build objects that subset images into groups,
        # these columns are removed below once their information is processed. These columns are, if they are not
        # among the chosen labels:
        # ["diseasestatus", patient_col]

        # Remove diseasestatus column that is only used to assign diseased and non-diseased index vectors from meta
        # data table:
        if "diseasestatus" not in label_selection:
            tissue_meta_data.drop("diseasestatus", 1, inplace=True)

        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {label: type for label, type in label_cols.items() if label in label_selection}
        label_tensors = {}
        label_names = {}  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "continuous":
                label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) / continuous_std[
                    feature]
                label_names[feature] = [feature]
        # 2. Scale percentages into [0, 1]
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "percentage":
                # Take "%" out of name if present
                feature_renamed = feature.replace("%", "percentage_")
                label_cols = dict([(k, v) if k != feature else (feature_renamed, v) for k, v in label_cols.items()])
                label_tensors[feature_renamed] = tissue_meta_data[feature].values / 100.
                label_names[feature_renamed] = [feature_renamed]
        # 3. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "categorical":
                tissue_meta_data[feature] = tissue_meta_data[feature].astype("str")
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "categorical":
                oh = pd.get_dummies(
                    tissue_meta_data[feature],
                    prefix=feature,
                    prefix_sep=">",
                    drop_first=False,
                    dtype=float,
                )
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array([i for i, x in enumerate(oh.columns) if x.endswith(">nan")])
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith(">nan")]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # 4. Add censoring information to survival
        # survival_mean = {
        #     feature: tissue_meta_data[feature].mean(skipna=True)
        #     for feature in list(label_cols.keys())
        #     if label_cols[feature] == "survival"
        # }
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "survival":
                if feature == "DFSmonth":
                    censor_col = "censor_DFS"
                if feature == "OSmonth":
                    censor_col = "censor_OS"
                label_tensors[feature] = np.concatenate([
                    np.expand_dims(tissue_meta_data[feature].values, axis=1),
                    np.expand_dims(tissue_meta_data[censor_col].values, axis=1),
                ], axis=1)
                label_names[feature] = [feature]
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        images = tissue_meta_data[img_key_col].values.tolist()
        assert np.all([len(list(self.img_to_patient_dict.keys())) == x.shape[0] for x in list(label_tensors.values())]), \
            "fatal processing error"
        label_tensors = {
            img: {
                kk: np.array(vv[images.index(img), :], ndmin=1)
                for kk, vv in label_tensors.items()  # iterate over labels
            } for img in self.img_to_patient_dict.keys()  # iterate over images
        }

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates



class DatasetMetabric(Dataset):
    """
    DatasetMetabric class. Inherits all functions from Dataset.
    Paper: Ali, H. R. et al. Imaging mass cytometry and multiplatform genomics define the phenogenomic landscape of breast cancer. Nat Cancer 1, 163–175 (2020).
    """

    cell_type_merge_dict = {
        "fine": {
            "B cells": "B cells",
            "Basal CKlow": "Tumor cells",
            "Endothelial": "Endothelial",
            "Fibroblasts": "Fibroblasts",
            "Fibroblasts CD68+": "Fibroblasts",
            "HER2+": "Tumor cells",
            "HR+ CK7-": "Tumor cells",
            "HR+ CK7- Ki67+": "Tumor cells",
            "HR+ CK7- Slug+": "Tumor cells",
            "HR- CK7+": "Tumor cells",
            "HR- CK7-": "Tumor cells",
            "HR- CKlow CK5+": "Tumor cells",
            "HR- Ki67+": "Tumor cells",
            "HRlow CKlow": "Tumor cells",
            "Hypoxia": "Tumor cells",
            "Macrophages Vim+ CD45low": "Macrophages",
            "Macrophages Vim+ Slug+": "Macrophages",
            "Macrophages Vim+ Slug-": "Macrophages",
            "Myoepithelial": "Myoepithelial",
            "Myofibroblasts": "Myofibroblasts",
            "T cells": "T cells",
            "Vascular SMA+": "Vascular SMA+"
        },
        "binary": {
            "B cells": "immune cells",
            "Basal CKlow": "other",
            "Endothelial": "other",
            "Fibroblasts": "other",
            "Fibroblasts CD68+": "other",
            "HER2+": "other",
            "HR+ CK7-": "other",
            "HR+ CK7- Ki67+": "other",
            "HR+ CK7- Slug+": "other",
            "HR- CK7+": "other",
            "HR- CK7-": "other",
            "HR- CKlow CK5+": "other",
            "HR- Ki67+": "other",
            "HRlow CKlow": "other",
            "Hypoxia": "other",
            "Macrophages Vim+ CD45low": "immune cells",
            "Macrophages Vim+ Slug+": "immune cells",
            "Macrophages Vim+ Slug-": "immune cells",
            "Myoepithelial": "other",
            "Myofibroblasts": "other",
            "T cells": "immune cells",
            "Vascular SMA+": "other"
        },
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""

        metadata = {
            "lateral_resolution": None,
            "fn": "single_cell_data/single_cell_data.csv",
            "image_col": "ImageNumber",
            "pos_cols": ["Location_Center_X", "Location_Center_Y"],
            "cluster_col": "description",
            "cluster_col_preprocessed": "description_preprocessed",
            "patient_col": "metabricId",
            "cell_type_coarseness": self.cell_type_coarseness,
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        cell_count = pd.DataFrame({"count": celldata_df.groupby(metadata["image_col"]).size()}).reset_index()
        img_ids_pass = set([
            x for x, y in zip(cell_count[metadata["image_col"]].values, cell_count["count"].values) if y >= 100
        ])
        celldata_df = celldata_df.iloc[np.where([x in img_ids_pass for x in celldata_df[metadata["image_col"]].values])[0], :]

        feature_cols = [
            "HH3_total",
            "CK19",
            "CK8_18",
            "Twist",
            "CD68",
            "CK14",
            "SMA",
            "Vimentin",
            "c_Myc",
            "HER2",
            "CD3",
            "HH3_ph",
            "Erk1_2",
            "Slug",
            "ER",
            "PR",
            "p53",
            "CD44",
            "EpCAM",
            "CD45",
            "GATA3",
            "CD20",
            "Beta_catenin",
            "CAIX",
            "E_cadherin",
            "Ki67",
            "EGFR",
            "pS6",
            "Sox9",
            "vWF_CD31",
            "pmTOR",
            "CK7",
            "panCK",
            "c_PARP_c_Casp3",
            "DNA1",
            "DNA2",
            "H3K27me3",
            "CK5",
            "Fibronectin"
        ]
        X = DataFrame(np.array(celldata_df[feature_cols]), columns=feature_cols)
        celldata = AnnData(
            X=X,
            obs=celldata_df[[metadata["image_col"], metadata["patient_col"], metadata["cluster_col"]]]
        )
        celldata.var_names_make_unique()

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype("category")

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {image key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.
        Parameters
        ----------
        label_selection
            Label selection.
        """
        patient_col = "METABRIC.ID"
        disease_features = {
            "grade": "categorical",
            "grade_collapsed": "categorical",
            "tumor_size": "continuous",
            "hist_type": "categorical",
            "stage": "categorical"
        }
        patient_features = {
            "age": "continuous",
            "menopausal": "categorical"
        }
        survival_features = {
            "time_last_seen": "survival"
        }
        tumor_features = {
            "ERstatus": "categorical",
            "lymph_pos": "continuous"

        }
        treatment_features = {
            "CT": "categorical",
            "HT": "categorical",
            "RT": "categorical",
            "surgery": "categorical",
            "NPI": "categorical"
        }
        col_renaming = {  # column aliases for convenience within this function
            "grade": "Grade",
            "tumor_size": "Size",
            "hist_type": "Histological.Type",
            "stage": "Stage",
            "age": "Age.At.Diagnosis",
            "menopausal": "Inferred.Menopausal.State",
            "death_breast": "DeathBreast",
            "time_last_seen": "T",
            "ERstatus": "ER.Status",
            "lymph_pos": "Lymph.Nodes.Positive",
            "surgery": "Breast.Surgery"
        }

        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)
        label_cols.update(survival_features)
        label_cols.update(tumor_features)
        label_cols.update(treatment_features)

        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(label_selection.intersection(set(list(label_cols.keys()))))
        label_cols_toread = [label for label in label_cols_toread if label != "grade_collapsed"]
        if "time_last_seen" in label_selection:
            censor_col = "death_breast"
            label_cols_toread = label_cols_toread + [censor_col]
        if "grade_collapsed" in label_selection and "grade" not in label_selection:
            label_cols_toread.append("grade")
        label_cols_toread_csv = [
            col_renaming[col] if col in list(col_renaming.keys()) else col
            for col in label_cols_toread
        ]

        usecols = label_cols_toread_csv + [patient_col] + ["Cohort", "Date.Of.Diagnosis"]
        tissue_meta_data = read_csv(
            os.path.join(self.data_path + "single_cell_data/41586_2019_1007_MOESM7_ESM.csv"),
            sep="\t",
            usecols=usecols
        )[usecols]
        tissue_meta_data.columns = label_cols_toread + [patient_col] + ["cohort", "date_of_diagnosis"]

        if "grade_collapsed" in label_selection:
            tissue_meta_data["grade_collapsed"] = ["3" if grade == "3" else "1&2" for grade in
                                                   tissue_meta_data["grade"]]
        if "grade_collapsed" in label_selection and "grade" not in label_selection:
            tissue_meta_data.drop("grade", 1, inplace=True)

        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {label: type for label, type in label_cols.items() if label in label_selection}
        label_tensors = {}
        label_names = {}  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "continuous":
                label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) \
                                         / continuous_std[feature]
                label_names[feature] = [feature]
        # 2. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "categorical":
                tissue_meta_data[feature] = tissue_meta_data[feature].astype("str")
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "categorical":
                oh = pd.get_dummies(
                    tissue_meta_data[feature],
                    prefix=feature,
                    prefix_sep=">",
                    drop_first=False,
                    dtype=float,
                )
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array([i for i, x in enumerate(oh.columns) if x.endswith(">nan")])
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith(">nan")]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # 3. Add censoring information to survival
        # survival_mean = {
        #     feature: tissue_meta_data[feature].mean(skipna=True)
        #     for feature in list(label_cols.keys())
        #     if label_cols[feature] == "survival"
        # }
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "survival":
                label_tensors[feature] = np.concatenate([
                    np.expand_dims(tissue_meta_data[feature].values, axis=1),
                    np.expand_dims(tissue_meta_data[censor_col].values, axis=1),
                ], axis=1)
                label_names[feature] = [feature]
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        tissue_meta_data_patients = tissue_meta_data[patient_col].values.tolist()
        label_tensors = {
            img: {
                feature_name: np.array(features[tissue_meta_data_patients.index(patient), :], ndmin=1)
                for feature_name, features in label_tensors.items()
            } if patient in tissue_meta_data_patients else None
            for img, patient in self.img_to_patient_dict.items()
        }
        # Reduce data to patients with graph-level labels:
        label_tensors = {k: v for k, v in label_tensors.items() if v is not None}
        self.img_celldata = {k: adata for k, adata in self.img_celldata.items() if k in label_tensors.keys()}

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates

class DatasetLung(Dataset):
    """DatasetLung class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "fine": {
            "Cancer": "tumor cells",
            "B cell": "B cells",
            "Neutrophils": "neutrophils",
            "NK cell": "NK cells",
            "DCs cell": "DC cells",
            "Endothelial cell": "endothelial",
            "Mast cell": "mast cells",
            "Tc": "cytotoxic T cells",
            "Th": "helper T cells",
            "Treg": "Tregs",
            "T other": "T cells",
            "Cl MAC": "macrophages",
            "Alt MAC": "macrophages",
            "Cl Mo": "monocytes",
            "Non-Cl Mo": "monocytes",
            "Int Mo": "monocytes",
            "undefined": "undefined",
        },
        "binary": {
            "Cancer": "other",
            "B cell": "immune cells",
            "Neutrophils": "immune cells",
            "NK cell": "immune cells",
            "DCs cell": "immune cells",
            "Endothelial cell": "other",
            "Mast cell": "immune cells",
            "Tc": "immune cells",
            "Th": "immune cells",
            "Treg": "immune cells",
            "T other": "immune cells",
            "Cl MAC": "immune cells",
            "Alt MAC": "immune cells",
            "Cl Mo": "immune cells",
            "Non-Cl Mo": "immune cells",
            "Int Mo": "immune cells",
            "undefined": "other",
        },
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": None,
            "fn": "single_cell_data.csv",
            "image_col": "sample",
            "pos_cols": ["x", "y"],
            "cluster_col": "celltype",
            "cluster_col_preprocessed": "celltype",
            "patient_col": "patient",
            "cell_type_coarseness": self.cell_type_coarseness,
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        feature_cols = []
        
        X = DataFrame(np.ones(celldata_df.shape))
        celldata = AnnData(X=X, obs=celldata_df[["sample", "patient", "celltype"]])
        celldata.var_names_make_unique()

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "category"
        )

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.
        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Graph features are based on TMA spot and not patient, thus patient_col is technically wrong.
        # For aspects where patients are needed (e.g. train-val-test split) the correct patients that are
        # loaded in _register_images() are used
        patient_col = "Key"
        # Sex (Male: 0, Female: 1)
        # Age (<75: 0, ≥75: 1)
        # BMI (<30: 0, ≥30: 1)
        # Smoking Status (Smoker: 0, Non-smoker:1)
        # Pack Years (1-30: 0, ≥30: 1)
        # Stage (I-II: 0, III-IV:1)
        # Progression (No: 0, Yes: 1)
        # Death (No: 0, Yes: 1)
        # Survival or loss to follow-up (years)
        # Predominant histological pattern (Lepidic:1, Papillary: 2, Acinar: 3, Micropapillary: 4, Solid: 5)
        disease_features = {}
        patient_features = {"Sex": "categorical", 
                            "Age": "categorical", 
                            "BMI": "categorical",
                            "Smoking Status": "categorical",
                            "Death": "categorical",
                           }
        survival_features = {
#             "Survival or loss to follow-up (years)": "survival"
        }
        tumor_features = {
            # not sure where these features belong
            "Stage": "categorical",
            "Progression": "categorical",
            "Predominant histological pattern": "categorical",
        }
        treatment_features = {}
        col_renaming = {}

        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)
        label_cols.update(survival_features)
        label_cols.update(tumor_features)
        label_cols.update(treatment_features)

        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(label_selection.intersection(set(list(label_cols.keys()))))
        
        label_cols_toread_csv = [
            col_renaming[col] if col in list(col_renaming.keys()) else col for col in label_cols_toread
        ]

        usecols = label_cols_toread_csv + [patient_col]
        tissue_meta_data = read_csv(
            os.path.join(self.data_path, "LUAD Clinical Data.csv"),
            sep=";",
            usecols=usecols,
        )[usecols]
        tissue_meta_data.columns = label_cols_toread + [patient_col]

        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {label: type for label, type in label_cols.items() if label in label_selection}
        label_tensors = {}
        label_names = {}  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "continuous":
                label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) / continuous_std[
                    feature
                ]
                label_names[feature] = [feature]
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "percentage":
                label_tensors[feature] = tissue_meta_data[feature]
                label_names[feature] = [feature]
        # 2. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "categorical":
                tissue_meta_data[feature] = tissue_meta_data[feature].astype("str")
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "categorical":
                oh = pd.get_dummies(tissue_meta_data[feature], prefix=feature, prefix_sep=">", drop_first=False, dtype=float)
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array([i for i, x in enumerate(oh.columns) if x.endswith(">nan")])
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.0)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith(">nan")]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # 3. Add censoring information to survival
        # survival_mean = {
        #     feature: tissue_meta_data[feature].mean(skipna=True)
        #     for feature in list(label_cols.keys())
        #     if label_cols[feature] == "survival"
        # }
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "survival":
                label_tensors[feature] = np.concatenate(
                    [
                        np.expand_dims(tissue_meta_data[feature].values, axis=1),
                        np.expand_dims(tissue_meta_data[censor_col].values, axis=1),
                    ],
                    axis=1,
                )
                label_names[feature] = [feature]
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        images = tissue_meta_data[patient_col].values.tolist()
        label_tensors = {
            img: {
                kk: np.array(vv[images.index(img), :], ndmin=1)
                for kk, vv in label_tensors.items()  # iterate over labels
            } for img in self.img_to_patient_dict.keys()  # iterate over images
        }

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates
