def change_type_graph_covariates(adata, label_list, dtype='float64'):
    """
    Adjust data type of graph covariates
    Args:
    adata (AnnData): anndata object
    label_list (List[str]): list of labels in graph covariates to change their type
    dtype (str, optional): desired type
        Defaults to float64
    """
    for label in label_list:
        adata.uns["graph_covariates"]["label_tensors"][label] = adata.uns["graph_covariates"]["label_tensors"][label]\
            .astype(dtype)
