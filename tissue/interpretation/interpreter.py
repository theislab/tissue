import pickle
from typing import Union

import torch 

import numpy as np

from tissue.estimators.base import Estimator
from tissue.data.loading import get_datamodule_from_curated
from tissue.data.datasets import DatasetBaselZurichZenodo, DatasetMetabric, DatasetSchuerch

KEY_LOCAL_ASSIGNMENT = None
KEY_LOCAL_LABEL = None
KEY_NODE_LABEL = None
DIM_NODE_LABEL = None

custom_palette = ["#4D5D67", "#4682B4", "#FFDB58"]


class Interpreter(Estimator):
    """
    Class for model interpretation. Inherits Estimator functionality and loads data
    and a model based on information saved with TrainModel.
    """

    def __init__(
            self,
            gs_path,
            gs_id,
            model_id,
            test_cv,
            val_cv,
            data_path,
            reload_raw=False,
            ss=False,
            cell_type_coarseness=None,
    ):
        """
        Parameters
        ----------
        results_path : Path to folder where model information was stored with TrainModel.
        model_id : Identifier given to model in TrainModel.
        model_class : Model class that was trained in TrainModel. Could be:
            - GCN
            - GCNSS
            - AGGCOND
            - SPATIAL
            - MI
            - MLP
        """
        self.results_path = gs_path + gs_id + "/results/"
        self.model_id = f'{model_id}_cv{test_cv}_{val_cv}'
        self.cell_type_coarseness = cell_type_coarseness

        runparams_file = self.results_path + model_id + '_runparams.pickle'
        with open(runparams_file, 'rb') as f:
            self.runparams = pickle.load(f)

        self._get_adata(data_path=data_path, reload_raw=reload_raw)
        self._reload_data(
            data_path=data_path,
            test_cv=test_cv,
            val_cv=val_cv,
            ss=ss,
            reload_raw=reload_raw,
        )
        self.model = None
        self._load_predictions()
        self.prepare_trainer()

        super().__init__(self.datamodule, self.model)

    def _get_adata(self, data_path, reload_raw):
        dataset = self.runparams['data_set']
        print(f"{dataset=}")
        print(f"{self.runparams['radius']=}")
        data_path = f'{data_path}{dataset}/'
        if dataset.lower() == "schuerch":
            Dataset = DatasetSchuerch
        elif dataset.lower() == "metabric":
            Dataset = DatasetMetabric
        elif dataset.lower() in ["jackson", "baselzurich", "zenodo"]:
            Dataset = DatasetBaselZurichZenodo
        else:
            raise ValueError("Dataset %r is not curated yet." % dataset)
        buffered_data_path = None if reload_raw else f"{data_path}buffer/"

        if self.cell_type_coarseness:
            self.adata = Dataset(
                data_path=data_path,
                buffered_data_path=buffered_data_path,
                radius=self.runparams['radius'],
                cell_type_coarseness=self.cell_type_coarseness,
            )
        else:
            self.adata = Dataset(
                data_path=data_path,
                buffered_data_path=buffered_data_path,
                radius=self.runparams['radius'],
                cell_type_coarseness=self.runparams['gs_id'].split('_')[4],
            )

        if dataset.lower() == 'metabric':  # filter out graphs without grade label
            exclude_indices = []
            img_celldata = {}
            img_to_patient_dict = {}
            for i, a in self.adata.img_celldata.items():
                if np.isnan(a.uns["graph_covariates"]["label_tensors"]["grade"]).any():
                    exclude_indices.append(int(i))
                else:
                    img_celldata[i] = a
                    img_to_patient_dict[i] = self.adata.img_to_patient_dict[i]
            celldata = self.adata.celldata[~self.adata.celldata.obs["ImageNumber"].isin(exclude_indices)]
            celldata.uns["img_to_patient_dict"] = img_to_patient_dict
            celldata.uns["img_keys"] = list(set(celldata.uns["img_keys"]) - set(exclude_indices))
            patients = np.array(list(img_to_patient_dict.values()))
            patients.sort()

            self.adata.img_celldata = img_celldata
            self.adata.celldata = celldata
            self.adata.img_to_patient_dict = img_to_patient_dict

        self.idx_to_key = {}
        self.cell_type = {}
        self.position = {}
        for i, img_key in enumerate(self.adata.img_celldata):
            self.idx_to_key[i] = img_key
            self.position[i] = self.adata.img_celldata[img_key].obsm["spatial"].astype(float)
            self.cell_type[i] = np.argmax(self.adata.img_celldata[img_key].obsm["node_types"], axis=1)

    def _reload_data(self, data_path, test_cv, val_cv, ss, reload_raw):

        data_set = self.runparams['data_set']

        feature_space = self.runparams['featur_space']
        if feature_space == 'molecular':
            key_x = "X"
        elif feature_space == 'celltype':
            key_x = "obsm/node_types"

        cell_type_coarseness = self.runparams['gs_id'].split('_')[4]
        print(f'infered cell type coarseness: {cell_type_coarseness}')

        num_workers = 4

        if data_set == 'schuerch':
            validation_split = 0.
        else:
            validation_split = 0.1
        test_split = 0.1

        if data_set == 'jackson':
            label_selection_target = ['grade']
        elif data_set == 'metabric':
            label_selection_target = ['grade']
        elif data_set == "schuerch":
            label_selection_target = ['Group']
        data_path = f'{data_path}{data_set}/'

        label_selection = self.runparams['graph_label_selection']
        KEY_GRAPH_LABEL = []
        if len(label_selection) > 1:
            for key in label_selection:
                KEY_GRAPH_LABEL.append(f"uns/graph_covariates/label_tensors/{key}")
        else:
            KEY_GRAPH_LABEL = f"uns/graph_covariates/label_tensors/{label_selection_target[0]}"

        buffered_data_path = None if reload_raw else f"{data_path}buffer/"
        self.datamodule = get_datamodule_from_curated(
            dataset=self.runparams['data_set'],
            data_path=data_path,
            buffered_data_path=buffered_data_path,
            radius=self.runparams['radius'],
            key_x=key_x,
            key_graph_supervision=KEY_GRAPH_LABEL,
            batch_size=self.runparams['batch_size'],
            cell_type_coarseness=cell_type_coarseness,
            edge_index=True,
            key_local_assignment=f"obsm/local_assignment" if ss else None,
            key_local_supervision=f"uns/graph_covariates/label_tensors/relative_cell_types" if ss else None,
            key_node_supervision=KEY_NODE_LABEL,
            num_workers=num_workers,
            preprocess=None,
            val_split=validation_split,
            test_split=test_split,
            seed_test=test_cv * 10 + test_cv,
            seed_val=val_cv * 10 + val_cv,
        )

    def _load_model(self):
        PATH = f'{self.results_path}{self.model_id}_model.pth'
        self.model = torch.load(PATH)

    def _load_predictions(self):
        with open(f'{self.results_path}{self.model_id}_predictions.pickle', 'rb') as f:
            preds = pickle.load(f)

        # concatenate batches in predictions
        predictions = {}
        for partition in preds.keys():
            predictions[partition] = {}
            try:
                for part in preds[partition]:
                    for key in part.keys():
                        try:
                            predictions[partition][key] = torch.concat((predictions[partition][key], part[key]))
                        except:
                            predictions[partition][key] = part[key]
            except:
                predictions[partition] = None
        self.predictions = predictions

    def _get_overall_gradients(
            self, 
            img_idx: Union[str, int],
            layer: str = "input",
            feature_space: str = "molecualr",
            return_neighborhood: bool = False,
    ):

        from tissue_pytorch.consts import BATCH_KEY_NODE_FEATURES

        if return_neighborhood:
            img_key = list(self.adata.img_celldata.keys())[img_idx]

            neighborhoods = self.adata.img_celldata[img_key].obsm["neighborhood"]
            neighborhood_names = self.adata.img_celldata[img_key].obsm["neighborhood_names"]

            dict_neighborhood_name = {}
            for i, n in enumerate(neighborhoods):
                if i in dict_neighborhood_name.keys():
                    continue
                dict_neighborhood_name[n] = neighborhood_names[i]


        self._load_model()

        dataloader = self.datamodule.predict_dataloader(idx=[img_idx])
        img_key = list(self.adata.img_celldata.keys())[img_idx]
        cell_types = self.adata.img_celldata[img_key].obsm["node_types"]

        activations = []
        def getActivation(name):
            def hook(model, input, output):
                activations.append(output)
            return hook
        
        self.model.node_embedding.layers[0].register_forward_hook(getActivation('node_emb'))
        self.model.use_local_supervision = False
        grads = []
        
        for d in dataloader:
            for batch in d:
                if layer == "input":
                    batch.x[BATCH_KEY_NODE_FEATURES].requires_grad = True
                    pred = self.model(batch)['graph_yhat']
                    g = []
                    for i in range(pred.shape[1]):
                        grad_outputs = torch.zeros_like(pred)
                        grad_outputs[:, i] = 1
                        grad = torch.autograd.grad(pred, batch.x[BATCH_KEY_NODE_FEATURES], grad_outputs=grad_outputs, retain_graph=True)[0].numpy()
                        g.append(grad)
                    g = np.array(g)  # outputs x nodes x features
                    if feature_space=="molecular":
                        g = g.mean(axis=2)  # outputs x nodes
                    else:
                        # select only for correct cell type
                        def reduce_dimension(matrix, one_hot_encoding):
                            # Apply einsum to reduce dimension based on one-hot encoding
                            reduced_matrix = torch.einsum('gnc,nc->gn', matrix, one_hot_encoding)

                            return reduced_matrix.numpy()
                        g = reduce_dimension(torch.Tensor(g), torch.Tensor(cell_types))

                    grads.append(g)
                else:
                    pred = self.model(batch)['graph_yhat']
                    g = []
                    for i in range(pred.shape[1]):
                        grad_outputs = torch.zeros_like(pred)
                        grad_outputs[:, i] = 1
                        grad = torch.autograd.grad(pred, activations[-1], grad_outputs=grad_outputs, retain_graph=True)[0].numpy()
                        g.append(grad)

                    g = np.array(g)  # outputs x nodes x filters
                    g = g.mean(axis=2)  # outputs x nodes
                    grads.append(g)
        grads = np.array(grads)  # batch x outputs x nodes
        gradients = np.sum(np.abs(grads.mean(axis=0)), axis=0)  # nodes
        
        
        mean = np.nanmean(np.where(gradients == 0, np.nan, gradients))
        std = np.nanstd(np.where(gradients == 0, np.nan, gradients))
        gradients = np.where(gradients == 0, 0, (gradients - mean) / std)

        if return_neighborhood:
            return gradients, neighborhoods, dict_neighborhood_name
        return gradients

    def plot_graph_embedding(
            self,
            embedding_method='pca',
            save: Union[str, None] = None,
            suffix: str = "_graphs.pdf",
            show: bool = True,
            data_key=None,
            return_embeddings: bool = False,
    ):
        """
        Plots a PCA or UMAP based on activations of one model layer as graph representations
        colored by a categorical graph label.

        Parameters
        ----------
        embedding_method : Either 'UMAP' or 'PCA'.
        save : Whether (if not None) and where (path as string given as save) to save plot.
        suffix : Suffix of file name to save to.
        show : Whether to display plot.

        Returns
        -------

        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import quantile_transform
        import scanpy as sc

        sns.set_style("whitegrid", {'axes.grid': False})

        plt.ioff()

        partitions = ['train', 'val', 'test']
        partitions = [p for p in partitions if self.predictions[p] is not None]
        part_name = np.concatenate([[p] * self.predictions[p]['graph_labels'].shape[0] for p in partitions])
        graph_embeddings = np.concatenate([self.predictions[p]['graph_embedding'] for p in partitions], axis=0)
        graph_embeddings = quantile_transform(graph_embeddings, output_distribution='uniform')

        labels = np.concatenate([self.predictions[p]['graph_labels'] for p in partitions], axis=0)
        grade = np.argmax(labels, axis=1) + 1

        obs = {'grade': grade,}

        graph_embeddings = sc.AnnData(graph_embeddings, obs=obs)

        embedding_method = embedding_method.lower()
        sc.pp.pca(graph_embeddings)
        if embedding_method == 'umap':
            sc.pp.neighbors(graph_embeddings)
            sc.tl.umap(graph_embeddings)

        if data_key == 'bz' or data_key == 'mb':
            palette = {
                1: custom_palette[0],
                2: custom_palette[1],
                3: custom_palette[2]
            }
            hue_order = [1, 2, 3]
        else:
            palette = {
                1: custom_palette[0],
                2: custom_palette[1],
            }
            hue_order = [1, 2]

        if embedding_method == 'umap':
            embedding = graph_embeddings.obsm['X_umap']
        elif embedding_method == 'pca':
            embedding = graph_embeddings.obsm['X_pca'][:, :2]
        hue = graph_embeddings.obs['grade']

        sizes = {p: 60. if p == 'test' else 30. for p in partitions}
        markers = {p: 'D' if p == 'test' else 'o' if p == 'train' else '^' for p in partitions}
        edgecolor = ['black' if p == 'test' else 'none' for p in part_name]

        g = sns.JointGrid(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=hue,
            hue_order=hue_order,
            palette=palette,
        )
        g.plot_joint(
            sns.scatterplot,
            size=part_name,
            sizes=sizes,
            style=part_name,
            markers=markers,
            edgecolor=edgecolor,
            linewidth=.5,
        )
        g.plot_marginals(
            sns.histplot,
            kde=True,
            alpha=0,
        )
        g.ax_joint.set_xticklabels([])
        g.ax_joint.set_yticklabels([])
        if embedding_method == 'pca':
            var1, var2 = graph_embeddings.uns['pca']['variance_ratio'][:2]
            # plt.xlabel(f'PC1 ({var1*100:.2f}%}')
            # plt.ylabel(f'PC2')
            g.ax_joint.set_xlabel(f'PC1 ({var1 * 100:.2f}%)')
            g.ax_joint.set_ylabel(f'PC2 ({var2 * 100:.2f}%)')

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.tight_layout()
            plt.savefig(f'{save}{embedding_method}{suffix}')
        if show:
            plt.show()
        plt.ion()
        if return_embeddings:
            return embedding, hue
        return None

    def plot_node_embedding(
            self,
            node_representation='input',  # 'input' or 'embedding'
            embedding_method='pca',  # 'pca' or 'umap'
            labels_to_plot=['image', 'grade'],
            return_adata: bool = False,
    ):
        """
        Plots a grid of UMAPs of node features / node feature embeddings after multiple layers of a model (rows)
        and overlayed with different characteristics (columns).

        Parameters
        ----------
        idx : Indices of graphs to use nodes from.
        layer_names : A list of layer names (check out model.summary()) for which UMAP of embedded nodes will be
                shown. Use 'input' for original node features. In that case loading data with get_data() is enough and
                no model is needed.
        plot_types : A list of characteristics that should be overlayed on the UMAP of each layer.
            - images: colors the nodes based on the image id
            - degree: indicates the node degree
            - a categorical graph/image label: indicates the label of the corresponding graph/image
        panel_width
        panel_height
        save : Whether (if not None) and where (path as string given as save) to save plot.
        suffix : Suffix of file name to save to.
        show : Whether to display plot.
        return_axs: Whether to return axis objects.

        Returns
        -------

        """
        import scanpy as sc

        from sklearn.preprocessing import quantile_transform

        if node_representation == 'input':
            node_representation_key = 'node_features'
        elif node_representation == 'embedding':
            node_representation_key = 'node_embedding'
        else:
            raise Exception("Choose 'input' or 'embedding' for argument node_representation")

        X = self.predictions['test'][node_representation_key].detach().numpy()

        image_id = np.concatenate([
            [i] * d.num_nodes
            for i, d in enumerate(self.datamodule.predict_dataloader(self.datamodule.idx_test).dataset)
        ]).astype(str)

        grade = np.concatenate([
            [grade] * d.num_nodes
            for grade, d in zip(np.argmax(self.predictions['test']['graph_labels'], axis=1) + 1,
                                self.datamodule.predict_dataloader(self.datamodule.idx_test).dataset)
        ]).astype(str)

        obs = {
            'image': image_id,
            'grade': grade,
        }

        node_embedding = sc.AnnData(X, obs=obs)

        # if node_representation == 'input':
        sc.pp.scale((node_embedding))
        # elif node_representation == 'embedding':
        #     node_embedding.X = quantile_transform(node_embedding.X, output_distribution='uniform')
        sc.pp.pca(node_embedding)
        if embedding_method == 'umap':
            # sc.pp.neighbors(node_embedding)
            # sc.tl.umap(node_embedding)
            import umap
            reducer = umap.UMAP(n_neighbors=15)
            embedding = reducer.fit_transform(node_embedding.X)
            node_embedding.obsm['X_umap'] = embedding

        if embedding_method == 'pca':
            sc.pl.pca(node_embedding, color=labels_to_plot, annotate_var_explained=True)
        elif embedding_method == 'umap':
            sc.pl.umap(node_embedding, color=labels_to_plot)
        else:
            raise Exception("Choose 'pca' or 'umap' for argument embedding_method")
        
        if return_adata:
            return node_embedding

    def plot_weight_matrix(
            self,
            target_label='Group',
            save: Union[str, None] = None,
            suffix: str = "_weight_matrix.pdf",
            panel_width: float = 4.,
            panel_height: float = 4.,
            show: bool = True,
    ):
        """
        Plots filter weights of the first GCN layer in relation to cell types
        together with the effect each filter has on the class prediction task.

        Model architecture this is meant for:
            - input: one-hot encoded cell types as node features
            - some node feature embedding layers 'assigning' one embedding vector to each cell type
            - one or more GCN layers
            - optionally any additional model elements...

        Assuming this model architecture, the first GCN layer becomes:
        A x H x E x W
        with:
        - the adjacency matrix A (#cells x #cells)
        - the one hot encoded cell type input matrix H (#cells x #cell_types)
        - the cell type embedding vectors E (#cell_types x emb_dim)
        - the weight matrix of the GCN layer W (emb_dim x #filters).

        By looking at E x W (#cell types x #filters) we can see what cell types and cell type combinations a filter
        is sensitive to. Additionally, we can check which effect a filter has on the final classification, by computing
        gradients from the model output wrt to the filter activations.

        :param layer_name: The layer from which the weight is taken.
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :return:
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.cm as cm
        import seaborn as sns
        sns.set_style("whitegrid", {'axes.grid': False})

        # get filter/weight matrix W (#cell types x #filters)

        self._load_model()

        for name, param in self.model.named_parameters():
            if name == 'node_embedding.layers.0.lin.weight' or name == 'node_embedding.layers.0.nn.0.weight'\
                    or name == 'node_embedding.layers.0.lin_src.weight':
                weights = param

        idx = self.datamodule.idx_test
        dataloader = self.datamodule.predict_dataloader(idx=idx)

        activations = []
        def getActivation(name):
            def hook(model, input, output):
                activations.append(output)
            return hook

        self.model.node_embedding.layers[0].register_forward_hook(getActivation('node_emb'))
        self.model.use_local_supervision = False
        grads = []
        for d in dataloader:
            for batch in d:
                pred = self.model(batch)['graph_yhat']
                g = []
                for i in range(pred.shape[1]):
                    grad_outputs = torch.zeros_like(pred)
                    grad_outputs[:, i] = 1
                    grad = torch.autograd.grad(pred, activations[-1], grad_outputs=grad_outputs, retain_graph=True)[0].numpy()
                    g.append(grad)
                g = np.array(g)  # outputs x nodes x filters
                g = g.mean(axis=1)  # outputs x filters
                grads.append(g)
        grads = np.array(grads)  # batch x outputs x filters
        gradients = grads.mean(axis=0)  # outputs x filters

        # group filters into the grades that they are most important for and sort them within that group by strength

        weights = weights.detach().numpy().T
        grade = np.argmax(gradients, axis=0)
        new_order = []
        for gr in np.unique(grade):
            filter = (grade != gr) * 10
            grad = gradients[gr] + filter
            indices = np.argsort(grad)[:np.sum(grade == gr)][::-1]
            new_order += list(indices)
        weights = weights[:, new_order]
        gradients = gradients[:, new_order]
        abs_weight = np.max(np.abs(weights))
        abs_grad = np.max(np.abs(gradients))

        n_filter = gradients.shape[1]
        cell_types = list(self.adata.celldata.uns['node_type_names'].values())
        class_labels = self.adata.celldata.uns['graph_covariates']['label_names'][target_label]
        class_labels = [' '.join(cl.split('>')) for cl in class_labels]

        # plot

        plt.ioff()
        fig, (ax_grads, ax_weights) = plt.subplots(2, 1, figsize=(panel_width, panel_height), gridspec_kw={'height_ratios': [3, 7]})

        im_pos = ax_weights.matshow(weights, cmap=cm.get_cmap('seismic'), vmin=-abs_weight, vmax=abs_weight)
        ax_weights.set_yticks(ticks=np.arange(0, len(cell_types), 1), labels=cell_types)
        ax_weights.set_xticks([], [])

        divider = make_axes_locatable(ax_weights)
        cax_weights = divider.append_axes("right", size='2%', pad=.2)
        fig.colorbar(im_pos, cax=cax_weights)

        im2 = ax_grads.matshow(gradients, cmap=cm.get_cmap('seismic'), vmin=-abs_grad, vmax=abs_grad)
        ax_grads.set_xticks(np.arange(n_filter), labels=np.arange(1, n_filter + 1))
        ax_grads.xaxis.tick_top()
        ax_grads.set_xlabel('filter')
        ax_grads.xaxis.set_label_position('top')
        ax_grads.set_yticklabels([''] + list(class_labels))

        divider = make_axes_locatable(ax_grads)
        cax_grads = divider.append_axes("right", size='2%', pad=.2)
        cbar = fig.colorbar(im2, cax=cax_grads)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)
        if show:
            plt.show()
        plt.close(fig)
        plt.ion()
        return None

    def plot_confusion_matrix(self, partition='test'):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        y_true = np.argmax(self.predictions[partition]['graph_labels'], axis=1)
        y_pred = np.argmax(self.predictions[partition]['graph_yhat'], axis=1)
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=np.arange(cm.shape[0]) + 1).plot()
        plt.grid(False)

    def plot_attention_matrix(
            self,
            panel_width: float = 4.,
            panel_height: float = 4.,
            save: Union[str, None] = None,
            suffix: str = "_attention_matrix.pdf",
            show: bool = True,
    ):

        import seaborn as sns
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        self._load_model()

        for name, param in self.model.named_parameters():
            if name == 'node_embedding.layers.0.att_src':
                att_src = param.detach().numpy().squeeze()
            if name == 'node_embedding.layers.0.att_dst':
                att_dst = param.detach().numpy().squeeze()
            if name == 'node_embedding.layers.0.lin_src.weight':
                weight = param.detach().numpy()

        att_ct_src = att_src @ weight
        att_ct_dst = att_dst @ weight

        if len(att_ct_src.shape) == 1:
            att = np.array([att_ct_src]).T + att_ct_dst
        elif len(att_ct_src.shape) == 2:
            att = att_ct_src.T @ att_ct_dst  # values before softmax

        if len(att_ct_src.shape) == 1:
            negative_slope = self.model.node_embedding.layers[0].negative_slope
            att = F.leaky_relu(torch.Tensor(att), negative_slope)

        att = att - np.max(att, axis=1, keepdims=True)  # softmax is computed after substraction of the maximum
        att = np.exp(att)

        att = att / att.max(axis=1, keepdims=True)

        cell_types = list(self.adata.celldata.uns['node_type_names'].values())

        plt.ioff()
        fig, ax = plt.subplots(1, 1, figsize=(panel_width, panel_height))
        sns.heatmap(att, cmap='viridis', cbar_kws={"shrink": .4})
        ax.set_box_aspect(1)
        plt.xticks(ticks=ax.get_xticks(), labels=cell_types, rotation=90)
        plt.yticks(ticks=ax.get_yticks(), labels=cell_types, rotation=0)
        plt.xlabel('neighbor cell type')
        plt.ylabel('source cell type')
        plt.grid(False)

        # Save, show and return figure.
        plt.tight_layout(rect=(0.15, 0, .9, 1))
        if save is not None:
            plt.savefig(f"{save}{suffix}")
        if show:
            plt.show()
        plt.close(fig)
        plt.ion()

    def plot_attention_filters(
            self,
            target_label='Group',
            panel_width: float = 4.,
            panel_height: float = 4.,
            save: Union[str, None] = None,
            suffix: str = "_attention_filter_matrix.pdf",
            show: bool = True,
    ):

        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.cm as cm
        import seaborn as sns
        sns.set_style("whitegrid", {'axes.grid': False})

        # get the attention matrix

        self._load_model()

        for name, param in self.model.named_parameters():
            if name == 'node_embedding.layers.0.att_src':
                att_src = param.detach().numpy().squeeze()
            if name == 'node_embedding.layers.0.att_dst':
                att_dst = param.detach().numpy().squeeze()
            if name == 'node_embedding.layers.0.lin_src.weight':
                weight = param.detach().numpy()

        att_ct_src = att_src @ weight
        att_ct_dst = att_dst @ weight

        if len(att_ct_src.shape) == 1:
            att = np.array([att_ct_src]).T + att_ct_dst
        elif len(att_ct_src.shape) == 2:
            att = att_ct_src.T @ att_ct_dst  # values before softmax

        if len(att_ct_src.shape) == 1:
            negative_slope = self.model.node_embedding.layers[0].negative_slope
            att = F.leaky_relu(torch.Tensor(att), negative_slope)

        att = att - np.max(att, axis=1, keepdims=True)  # softmax is computed after substraction of the maximum
        att = np.exp(att)

        att = att / att.max(axis=1, keepdims=True)

        # do the filter plots

        # get filter/weight matrix W (#cell types x #filters)

        for name, param in self.model.named_parameters():
            if name == 'node_embedding.layers.0.lin.weight' or name == 'node_embedding.layers.0.nn.0.weight' \
                    or name == 'node_embedding.layers.0.lin_src.weight':
                weights = param

        idx = self.datamodule.idx_test
        dataloader = self.datamodule.predict_dataloader(idx=idx)

        activations = []

        def getActivation(name):
            def hook(model, input, output):
                activations.append(output)

            return hook

        self.model.node_embedding.layers[0].register_forward_hook(getActivation('node_emb'))
        self.model.use_local_supervision = False
        grads = []
        for d in dataloader:
            for batch in d:
                pred = self.model(batch)['graph_yhat']
                g = []
                for i in range(pred.shape[1]):
                    grad_outputs = torch.zeros_like(pred)
                    grad_outputs[:, i] = 1
                    grad = torch.autograd.grad(pred, activations[-1], grad_outputs=grad_outputs, retain_graph=True)[
                        0].numpy()
                    g.append(grad)
                g = np.array(g)  # outputs x nodes x filters
                cts = batch.x['node_features'].numpy()  # nodes x cell_types
                cts = cts / cts.sum(axis=0, keepdims=True)
                # cts[cts != cts] = 0
                a = np.einsum('abc,bd->acd', g, cts)  # outputs x filters x cell_types
                grads.append(a)
        grads = np.array(grads)  # batch x outputs x filters x cell_types
        gradients = np.nanmean(grads, axis=0)  # outputs x filters x cell_types

        gradients_sum = gradients.sum(axis=-1)

        # group filters into the grades that they are most important for and sort them within that group by strength

        weights = weights.detach().numpy().T
        grade = np.argmax(gradients_sum, axis=0)
        new_order = []
        for gr in np.unique(grade):
            filter = (grade != gr) * 10
            grad = gradients_sum[gr] + filter
            indices = np.argsort(grad)[:np.sum(grade == gr)][::-1]
            new_order += list(indices)
        weights = weights[:, new_order]
        gradients = gradients[:, new_order, :]
        abs_weight = np.max(np.abs(weights))
        abs_grad = np.max(np.abs(gradients))

        n_filter = gradients.shape[1]
        cell_types = list(self.adata.celldata.uns['node_type_names'].values())
        class_labels = self.adata.celldata.uns['graph_covariates']['label_names'][target_label]
        class_labels = [' '.join(cl.split('>')) for cl in class_labels]

        # plot

        for i, ct in enumerate(cell_types):
            print(ct)
            att_vector = att[i]

            weights_ct = weights * np.array([att_vector]).T

            plt.ioff()
            fig, (ax_grads, ax_weights) = plt.subplots(2, 1, figsize=(panel_width, panel_height),
                                                       gridspec_kw={'height_ratios': [3, 7]})

            im_pos = ax_weights.matshow(weights_ct, cmap=cm.get_cmap('seismic'), vmin=-abs_weight, vmax=abs_weight)
            ax_weights.set_yticks(ticks=np.arange(0, len(cell_types), 1), labels=cell_types)
            ax_weights.set_xticks([], [])

            divider = make_axes_locatable(ax_weights)
            cax_weights = divider.append_axes("right", size='2%', pad=.2)
            fig.colorbar(im_pos, cax=cax_weights)

            im2 = ax_grads.matshow(gradients[:, :, i], cmap=cm.get_cmap('seismic'), vmin=-abs_grad, vmax=abs_grad)
            ax_grads.set_xticks(np.arange(n_filter), labels=np.arange(1, n_filter + 1))
            ax_grads.xaxis.tick_top()
            ax_grads.set_xlabel('filter')
            ax_grads.xaxis.set_label_position('top')
            ax_grads.set_yticklabels([''] + list(class_labels))

            divider = make_axes_locatable(ax_grads)
            cax_grads = divider.append_axes("right", size='2%', pad=.2)
            cbar = fig.colorbar(im2, cax=cax_grads)

            # Save, show and return figure.
            plt.tight_layout(rect=(0.15, 0, .9, 1))
            if save is not None:
                plt.savefig(f"{save}_{ct}{suffix}")
            if show:
                plt.show()
            plt.close(fig)
            plt.ion()
    
    def plot_graph(
        self,
        img_idx,
        save: Union[str, None] = None,
        suffix: str = "_graphs.png",
        show: bool = True,
        return_axs: bool = False,
    ): 
        """Plot graphs with cell type colored nodes"""

        from torch_geometric.utils import to_networkx
        import matplotlib.pyplot as plt
        import networkx as nx
        import matplotlib.cm as cmx
        import matplotlib.colors as colors

        from tissue.consts import BATCH_KEY_NODE_FEATURES

        plt.ioff()

        if isinstance(img_idx, str) or isinstance(img_idx, int):

            img_key = list(self.adata.img_celldata.keys())[img_idx]

            data = self.datamodule.data[img_idx]

            node_features = data.x[BATCH_KEY_NODE_FEATURES]
            #edge_indices = data.edge_index

            cell_types = self.cell_type[img_idx]

            if cell_types is None:
                raise ValueError('No cell types for this image available')


            cell_type_names = {i: value 
                   for i, value in enumerate(list(self.adata.img_celldata[img_key].uns["node_type_names"].values()))}
            
            vmax = len(cell_type_names)
            cNorm = colors.Normalize(vmin=0, vmax=vmax)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('tab20c'))
            
            graph = to_networkx(data, to_undirected=True)

            pos_nodes = {node_idx: self.position[img_idx][node_idx] for node_idx, _ in enumerate(node_features)}

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            nx.draw_networkx_edges(
                graph,
                pos=pos_nodes,
                width=0.3
            )

            for ctype in np.unique(cell_types):
                color = [scalarMap.to_rgba(ctype)]
                idx_c = list(np.where(cell_types == ctype)[0])
                nx.draw_networkx_nodes(
                    graph,
                    node_size=10,
                    nodelist=idx_c,
                    node_color=color,
                    pos=pos_nodes,
                    label=cell_type_names[ctype]
                )
            lgd = fig.legend(bbox_to_anchor=(1.45, 0.9))
            plt.grid(False)
            plt.gca().invert_yaxis()

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None
        
    def plot_gradient_on_graph(
            self, 
            img_idx: Union[str, int],
            save: Union[str, None] = None,
            suffix: str = "_gradient_graph.pdf",
            show: bool = True,
            return_axs: bool = False,
            layer: str = "input", #[node_embedding, input]
            feature_space: str = "molecular",
    ):
        import matplotlib.pyplot as plt
        import networkx as nx
        import matplotlib.cm as cmx
        import matplotlib.colors as colors
        import scipy

        from torch_geometric.utils import to_networkx

        from tissue_pytorch.consts import BATCH_KEY_NODE_FEATURES

        self._load_model()

        dataloader = self.datamodule.predict_dataloader(idx=[img_idx])
        img_key = list(self.adata.img_celldata.keys())[img_idx]
        cell_types = self.adata.img_celldata[img_key].obsm["node_types"]

        activations = []
        def getActivation(name):
            def hook(model, input, output):
                activations.append(output)
            return hook

        self.model.node_embedding.layers[0].register_forward_hook(getActivation('node_emb'))
        self.model.use_local_supervision = False
        grads = []
        for d in dataloader:
            for batch in d:
                if layer == "input":
                    batch.x[BATCH_KEY_NODE_FEATURES].requires_grad = True
                    pred = self.model(batch)['graph_yhat']
                    g = []
                    for i in range(pred.shape[1]):
                        grad_outputs = torch.zeros_like(pred)
                        grad_outputs[:, i] = 1
                        grad = torch.autograd.grad(pred, batch.x[BATCH_KEY_NODE_FEATURES], grad_outputs=grad_outputs, retain_graph=True)[0].numpy()
                        g.append(grad)
                else:
                    pred = self.model(batch)['graph_yhat']
                    g = []
                    for i in range(pred.shape[1]):
                        grad_outputs = torch.zeros_like(pred)
                        grad_outputs[:, i] = 1
                        grad = torch.autograd.grad(pred, activations[-1], grad_outputs=grad_outputs, retain_graph=True)[0].numpy()
                        g.append(grad)
                if feature_space == "molecular":
                    g = np.array(g)  # outputs x nodes x features
                    g = g.mean(axis=2)  # outputs x nodes
                else:
                    g = np.array(g)  # outputs x nodes x features

                    # select only for correct cell type
                    def reduce_dimension(matrix, one_hot_encoding):
                        # Apply einsum to reduce dimension based on one-hot encoding
                        reduced_matrix = torch.einsum('gnc,nc->gn', matrix, one_hot_encoding)

                        return reduced_matrix.numpy()
                    g = reduce_dimension(torch.Tensor(g), torch.Tensor(cell_types))

                grads.append(g)
        grads = np.array(grads)  # batch x outputs x nodes
        gradients = np.abs(grads.mean(axis=0))  # outputs x nodes
        

        # only keep gradient for class where it is maximal
        #max_grad = np.max(gradients, axis=0)
        #argmax_grad = np.argmax(gradients, axis=0)

        n_outputs = len(gradients)

        #grad = np.array([max_grad] * n_outputs)
        #for i in range(n_outputs):
        #    grad[i, argmax_grad != i] = 0
        
        grad = gradients
        mean = np.nanmean(np.where(grad == 0, np.nan, grad))
        std = np.nanstd(np.where(grad == 0, np.nan, grad))
        grad = np.where(grad == 0, 0, (grad - mean) / std)


        fig = plt.figure(figsize=(10 * n_outputs, 7))

        img_key = list(self.adata.img_celldata.keys())[img_idx]

        data = self.datamodule.data[img_idx]

        node_features = data.x[BATCH_KEY_NODE_FEATURES]
        #edge_indices = data.edge_index

        cell_types = self.cell_type[img_idx]

        if cell_types is None:
            raise ValueError('No cell types for this image available')


        cell_type_names = {i: value 
                for i, value in enumerate(list(self.adata.img_celldata[img_key].uns["node_type_names"].values()))}
            
        vmax = len(cell_type_names)
        cNorm = colors.Normalize(vmin=0, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('tab20c'))

        pos_nodes = {node_idx: self.position[img_idx][node_idx] for node_idx, _ in enumerate(node_features)}


        for i in range(n_outputs):
            ax = fig.add_subplot(1, n_outputs, i+1)
            graph = to_networkx(data,  to_undirected=True)

            #width = scipy.sparse.csr_matrix(np.abs(grad[i])).data 
            
            width = np.abs(grad[i])
            
            width = width #/ np.max(width)
           
            nx.draw_networkx_edges(
                graph,
                pos=pos_nodes,
                width=width,
            )
            if isinstance(width, int) and width == 0:
                node_alpha = 0

            else:
                node_alpha =  np.abs(width) 
                node_alpha = node_alpha / np.max(node_alpha)
            """for ctype in np.unique(cell_types):
                color = [scalarMap.to_rgba(ctype)]
                idx_c = list(np.where(cell_types == ctype)[0])
                nx.draw_networkx_nodes(
                    graph,
                    node_size=8,
                    nodelist=idx_c,
                    node_color=color,
                    pos=pos_nodes,
                    #alpha=node_alpha,
                    label=cell_type_names[ctype]
                )"""
            try:
                width = grad[i]
                nodes = nx.draw_networkx_nodes(
                        graph,
                        node_size=8,
                        #nodelist=idx_c,
                        #node_color=color,
                        cmap=plt.get_cmap('coolwarm'), 
                        node_color=width,
                        pos=pos_nodes,
                        #alpha=node_alpha,
                        #label=cell_type_names[ctype]
                    )
                plt.gca().invert_yaxis()
                cmin = np.min(grad)
                cmax = np.max(grad)
                c = max(abs(cmax), abs(cmin)) 
                import matplotlib as mpl
                plt.colorbar(nodes)
                nodes.set_clim(-c, c)
                
            except:
                continue

            if i == n_outputs - 1:
                lgd = ax.legend(bbox_to_anchor=(1.0, 1.0))
                

            plt.grid(False)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix,  bbox_inches='tight', dpi=300)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def plot_overall_gradient_on_graph(
            self, 
            img_idx: Union[str, int],
            save: Union[str, None] = None,
            suffix: str = "_gradient_graph.pdf",
            show: bool = True,
            return_axs: bool = False,
            layer: str = "input",
            feature_space: str = "molecualr",
            return_gradients: bool = False,
    ):
        import matplotlib.pyplot as plt
        import networkx as nx
        import matplotlib.cm as cmx
        import matplotlib.colors as colors
        import scipy

        from torch_geometric.utils import to_networkx

        from tissue_pytorch.consts import BATCH_KEY_NODE_FEATURES

        gradients = self._get_overall_gradients(img_idx=img_idx, layer=layer, feature_space=feature_space)
        fig = plt.figure(figsize=(7, 7))

        img_key = list(self.adata.img_celldata.keys())[img_idx]

        data = self.datamodule.data[img_idx]

        node_features = data.x[BATCH_KEY_NODE_FEATURES]
        #edge_indices = data.edge_index
        pos_nodes = {node_idx: self.position[img_idx][node_idx] for node_idx, _ in enumerate(node_features)}


        cell_types = self.cell_type[img_idx]


        graph = to_networkx(data,  to_undirected=True)

        ax = fig.add_subplot(111)
        nx.draw_networkx_edges(
            graph,
            pos=pos_nodes,
            width=0.3
        )
        sc = nx.draw_networkx_nodes(
            graph,     
            node_size=8,
            pos=pos_nodes,
            node_color=gradients,
            cmap=plt.get_cmap('coolwarm'),
            #label=gradients,
        )
        
        c = max(np.abs(np.max(gradients)), np.abs(np.min(gradients)))
        plt.gca().invert_yaxis()
        plt.grid(False)
        sc.set_clim(-c, c)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax_pos = divider.append_axes("right", size='5%', pad=0.08)
        fig.colorbar(sc, ax=ax, cax=cax_pos)

        # Save, show and return figure.
        plt.tight_layout()
        plt.grid(False)
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None
