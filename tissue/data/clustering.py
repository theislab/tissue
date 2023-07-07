import numpy as np

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph

from tissue.imports.transforms import get_adjacency_from_adata


def prepare_spectral_clusters(  
    adata,
    n_cluster,
    k_neighbors=10,
):
    """
    Computes spectral clusterings for adata

    Parameters
    ----------
    adata
        anndata object
    n_cluster : int
        The number of spectral clusters to be produced for the self-supervision task.
    k_neighbors : int
        The number of neighbors used of the knn graph construction.

    Returns
    -------
    node_to_cluster_mapping
        One hot encoded matrix (n_nodes, n_clusters) assigning graph nodes
        to their cluster.
    within_cluster_a
        Transformed adjacency matrices with edges between clusters removed.
    between_cluster_a
        Adjacency matrices describing the connectivity of the clusters.

    """

    # Compute knn matrix
    knn_matrix = kneighbors_graph(
            adata.obsm["spatial"],
            n_neighbors=k_neighbors,
            mode="connectivity",  # also "distance" possible
            include_self=True
        )

    # Compute spectral clusters and one-hot encoded assignments from graph nodes to clusters
    clusterer = SpectralClustering(
        n_clusters=n_cluster,
        affinity="precomputed",
    )

    def to_one_hot(a):
        res = np.zeros((len(a), np.max(a) + 1))
        res[np.arange(len(a)), a] = 1
        return res

    node_to_cluster_mapping = to_one_hot(clusterer.fit_predict(X=knn_matrix))

    adj_matrix = get_adjacency_from_adata(adata)

    # Compute adjacency matrices containing only within-cluster edges
    within_cluster_a = adj_matrix.multiply(node_to_cluster_mapping @ np.transpose(node_to_cluster_mapping))

    # Compute connectivity of clusters
    between_cluster_a = (np.transpose(node_to_cluster_mapping) @ knn_matrix @ node_to_cluster_mapping > 0)\
                            .astype(float) - np.eye(node_to_cluster_mapping.shape[1])

    return node_to_cluster_mapping, within_cluster_a, between_cluster_a
    

def get_self_supervision_label(
    adata,
    label,
    n_clusters,
    k_neighbors=10,
):
    """
    Computes a label per cluster used for a self-supervision task. This is usually some form of description of the
    surrounding of a cluster.

    Parameters
    ----------
    adata: anndata object
    label : str
        Name of the supervision label to be prepared. Valid options are:

        - "relative_cell_types" - the cell type frequency of all clusters connected to one cluster.
    n_cluster: int
        Number of spectral clusters
    k_neighbours: int
        Number neighbors used for knn graph construction
    Returns
    -------
    Matrix (n_clusters, n_types) containing the cell type frequencies of all nodes
    within clusters connected to one cluster for all the cluster.
    """

    node_to_cluster_mapping, within_cluster_a, between_cluster_a = prepare_spectral_clusters(
        adata=adata,
        n_cluster=n_clusters,
        k_neighbors=k_neighbors,
    )

    if label == "relative_cell_types":
        surrounding_cell_types = between_cluster_a @ np.transpose(node_to_cluster_mapping) @ adata.obsm["node_types"]
            
        rel_cell_types = surrounding_cell_types / np.maximum(np.sum(surrounding_cell_types, axis=1, keepdims=True),
                                                             np.ones((surrounding_cell_types.shape[0], 1)))
        return rel_cell_types, node_to_cluster_mapping
    else:
        raise ValueError(f"Self-supervision label {label} not recognized")
