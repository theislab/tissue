"""
Note on position of tensors in Data as used in .forward():

All models described here use the same (spatial) data loader to allow full comparability.
The tensors are structured as follows:

    node features: data.x[BATCH_KEY_NODE_FEATURES]  # (batch, nodes in graph, node features)
    node labels: data.x[BATCH_KEY_NODE_LABELS]  # (batch, nodes in graph, node labels)
    graph labels: data.y[BATCH_KEY_GRAPH_LABELS]  # (batch, graph labels)

Note data.y is used for graph-level features. Therefore, all putative node-level features are included in data.x,
which is split in forward() into features and labels.
"""

from typing import List, Union

import torch
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from tissue.consts import BATCH_KEY_GRAPH_LABELS, BATCH_KEY_GRAPH_EMBEDDING, BATCH_KEY_GRAPH_YHAT, \
    BATCH_KEY_LOCAL_ASSIGNMENTS, BATCH_KEY_LOCAL_LABELS, BATCH_KEY_LOCAL_YHAT, BATCH_KEY_NODE_EMBEDDING, \
    BATCH_KEY_NODE_FEATURES, BATCH_KEY_NODE_FEATURE_EMBEDDING, BATCH_KEY_NODE_LABELS, BATCH_KEY_NODE_YHAT
from tissue.models.modules.gnn import ModuleGCN, ModuleGIN, ModuleGAT, ModuleGATInteraction
from tissue.models.modules.mlp import ModuleMLP


class SampleEmbeddingBase(pl.LightningModule):
    use_node_supervision: bool

    def __init__(self, dim_features: int, 
                 dim_groups_local: Union[None, int] = None, 
                 dim_node_types: int = None,
                 use_node_supervision: bool = False, 
                 type_graph_label: Union[List[str], str] = "categorical",
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)

        self.loss_fn = torch.nn.MSELoss(reduction="mean")

        # for multi-task learning
        print(f"{type_graph_label=}")
        if isinstance(type_graph_label, list):
            self.graph_loss_fn = []
            for t in type_graph_label:
                if t == "categorical":
                    self.graph_loss_fn.append(torch.nn.CrossEntropyLoss(reduction="mean"))
                elif t == "continuous":
                    self.graph_loss_fn.append(torch.nn.MSELoss())
                elif t == "percentage":
                    self.graph_loss_fn.append(torch.nn.BCEWithLogitsLoss())
                elif t == "survival":
                    self.graph_loss_fn.append(torch.nn.L1Loss())
                else:
                    raise NotImplementedError
        else:
            self.graph_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.node_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.local_loss_fn = torch.nn.MSELoss(reduction='mean')
        self.dim_features = dim_features
        self.dim_groups_local = dim_groups_local
        self.dim_node_types = dim_node_types
        self.use_local_supervision = dim_groups_local is not None
        self.use_node_supervision = use_node_supervision

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, "min",
            factor=self.lr_schedule_factor,
            patience=self.lr_schedule_patience,
            min_lr=self.lr_schedule_min_lr
        )
        optim = {"optimizer": optim,
                 "lr_scheduler": sch, 
                 "monitor": "train_loss"}
        return optim

    def step_to_loss(self, batch, batch_idx, mode):
        batch_size = 1
        if type(batch) == list:
            batch_size = len(batch)
        x = self.step(batch=batch, batch_idx=batch_idx)
        recon_loss = self.loss(fwd_pass=x, batch_size=batch_size)
        return recon_loss, batch_size
    
    def step(self, batch, batch_idx):
        if type(batch) == list:
            batch = Batch.from_data_list(batch)
        fwd_pass = self(batch)
        return fwd_pass

    def loss_graph(self, fwd_pass):
        # for multi-task learning - can be replaced by if self.multi_task
        if isinstance(fwd_pass[BATCH_KEY_GRAPH_LABELS], dict):
            loss = 0
            for loss_fn in self.graph_loss_fn:
                for i, l in enumerate(fwd_pass[BATCH_KEY_GRAPH_LABELS]):
                    loss += loss_fn(fwd_pass[BATCH_KEY_GRAPH_YHAT][i], fwd_pass[BATCH_KEY_GRAPH_LABELS][l])
            return loss
        
        return self.graph_loss_fn(fwd_pass[BATCH_KEY_GRAPH_YHAT], fwd_pass[BATCH_KEY_GRAPH_LABELS])

    def loss_local(self, fwd_pass):
        return self.local_loss_fn(fwd_pass[BATCH_KEY_LOCAL_YHAT], fwd_pass[BATCH_KEY_LOCAL_LABELS])

    def loss_node(self, fwd_pass):
        return self.node_loss_fn(fwd_pass[BATCH_KEY_NODE_YHAT], fwd_pass[BATCH_KEY_NODE_LABELS])

    def loss(self, fwd_pass, batch_size):
        loss = self.loss_graph(fwd_pass=fwd_pass)
        if self.use_local_supervision:
            loss += self.loss_local(fwd_pass=fwd_pass)
        if self.use_node_supervision:
            loss += self.loss_node(fwd_pass=fwd_pass)
        return loss

    def training_step(self, data_list, batch_idx):
        loss, batch_size = self.step_to_loss(data_list, batch_idx, "train")
        # self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, data_list, batch_idx):
        loss, batch_size = self.step_to_loss(data_list, batch_idx, "val")
        self.log("val_loss", loss, batch_size=batch_size, prog_bar=True)

    def test_step(self, data_list, batch_idx):
        loss, batch_size = self.step_to_loss(data_list, batch_idx, "test")
        self.log("test_loss", loss, batch_size=batch_size, prog_bar=True)
    
    def predict_step(self, data_list, batch_idx):
        fwd_pass = self.step(data_list, batch_idx)
        return fwd_pass


class GraphEmbedding(SampleEmbeddingBase):
    def __init__(
            self,
            dim_features: int,
            dim_graph_label: int,
            dim_node_embedding: List[int],
            dim_mlp: List[int],
            encode_features: bool = True,
            activation: torch.nn.Module = torch.nn.LeakyReLU(0.1),
            dim_local_label: Union[None, int] = None,
            dim_node_label: Union[None, int] = None,
            type_graph_label: str = "categorical",
            final_pool: str = "mean",  # mean, max, sum
            **kwargs
    ):
        super().__init__(dim_features=dim_features, type_graph_label=type_graph_label, **kwargs)
        self.dim_latent = dim_node_embedding[-1]
        self.dim_graph_label = dim_graph_label
        self.dim_local_label = dim_local_label
        self.encode_features = encode_features
        self.type_graph_label = type_graph_label
        self.multi_task = False

        # add feature embedding on node-level
        if self.encode_features:
            self.feature_embedding = ModuleMLP(activation=activation, in_channels=dim_features,
                                               units=dim_node_embedding)
            in_channels = self.dim_latent
        else:
            in_channels = dim_features
        # node-wise GCN pass
        self.node_embedding = ModuleGCN(activation=activation, in_channels=in_channels, units=dim_node_embedding)
        
        # graph-level supervision
        # allow for multi-task learning
        if isinstance(dim_graph_label, list):
            self.multi_task = True
            self.graph_clf = torch.nn.ModuleList([ModuleMLP(activation=activation, in_channels=self.dim_latent,
                                                            units=dim_mlp + [dim_graph_label[i]], 
                                                            type_graph_label=type_graph_label[i],
                                                            prediction_transformation=True) 
                                                            for i in range(len(dim_graph_label))])
        else:
            self.graph_clf = ModuleMLP(activation=activation, in_channels=self.dim_latent,
                                    units=dim_mlp + [dim_graph_label],
                                    type_graph_label=type_graph_label,
                                    prediction_transformation=True)
        # node-level supervision
        if self.use_node_supervision:
            self.node_clf = torch.nn.Linear(in_features=self.dim_latent, out_features=dim_node_label)
        # self-supervision
        if self.use_local_supervision:
            self.local_clf = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.dim_latent, out_features=dim_local_label),
                torch.nn.Softmax(dim=1),
            )

        # defining node pooling function
        if final_pool == "mean":
            self.pool_fn = global_mean_pool
        elif final_pool == "max":
            self.pool_fn = global_max_pool
        elif final_pool == "sum":
            self.pool_fn = global_add_pool
        else:
            ValueError("Final pool method must be on of ['max', 'mean', 'sum']")

    def forward(self, data):
        x = {
            BATCH_KEY_NODE_FEATURES: data.x[BATCH_KEY_NODE_FEATURES],
            BATCH_KEY_GRAPH_LABELS: data.y[BATCH_KEY_GRAPH_LABELS],
        }
        if BATCH_KEY_NODE_LABELS in data.y.keys():
            x[BATCH_KEY_NODE_LABELS] = data.x[BATCH_KEY_NODE_LABELS]

        if BATCH_KEY_LOCAL_ASSIGNMENTS in data.x.keys():
            x[BATCH_KEY_LOCAL_ASSIGNMENTS] = data.x[BATCH_KEY_LOCAL_ASSIGNMENTS]
            x[BATCH_KEY_LOCAL_LABELS] = data.y[BATCH_KEY_LOCAL_LABELS]

        if self.encode_features:
            # embed features on node-level
            x[BATCH_KEY_NODE_FEATURE_EMBEDDING] = self.feature_embedding(x[BATCH_KEY_NODE_FEATURES])
            input_key = BATCH_KEY_NODE_FEATURE_EMBEDDING
        else:
            input_key = BATCH_KEY_NODE_FEATURES

        batch = data.batch if "batch" in data.keys else None  # defining nodes batch

        # forward pass of node-wise GCN
        x[BATCH_KEY_NODE_EMBEDDING] = self.node_embedding(x=x[input_key], edge_index=data.edge_index)
        if self.use_node_supervision:
            x[BATCH_KEY_NODE_YHAT] = self.node_clf(x[BATCH_KEY_NODE_EMBEDDING])

        if self.use_local_supervision:
            # define local assignment per batch for each node
            if not batch is None:
                batch_local_assignment = batch * self.dim_groups_local + x[BATCH_KEY_LOCAL_ASSIGNMENTS]
                batch_local_assignment = batch_local_assignment.to(torch.int64)
            else:
                batch_local_assignment = x[BATCH_KEY_LOCAL_ASSIGNMENTS]
            z = global_mean_pool(x[BATCH_KEY_NODE_EMBEDDING], batch=batch_local_assignment)
            x[BATCH_KEY_LOCAL_YHAT] = self.local_clf(z)

        # Node pooling to obtain graph embedding
        x[BATCH_KEY_GRAPH_EMBEDDING] = self.pool_fn(x[BATCH_KEY_NODE_EMBEDDING], batch=batch)
        # reshape graph labels to (n_graphs, dim_graph_label)
        try:
            num_graphs = data._num_graphs
        except:
            num_graphs = 1

        if self.multi_task:
            for i, l in enumerate(x[BATCH_KEY_GRAPH_LABELS].keys()):
                x[BATCH_KEY_GRAPH_LABELS][l] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS][l], (num_graphs, self.dim_graph_label[i]))
        else:
            x[BATCH_KEY_GRAPH_LABELS] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS], (num_graphs, self.dim_graph_label))

        # graph level supervision (FC)
        if self.multi_task:
            x[BATCH_KEY_GRAPH_YHAT] = [clf(x[BATCH_KEY_GRAPH_EMBEDDING]) for clf in self.graph_clf]
        else:
            x[BATCH_KEY_GRAPH_YHAT] = self.graph_clf(x[BATCH_KEY_GRAPH_EMBEDDING])
        return x


class GraphAttentionEmbedding(SampleEmbeddingBase):
    def __init__(
            self,
            dim_features: int,
            dim_graph_label: int,
            dim_node_embedding: List[int],
            dim_mlp: List[int],
            encode_features: bool = True,
            activation: torch.nn.Module = torch.nn.LeakyReLU(0.1),
            dim_local_label: Union[None, int] = None,
            dim_node_label: Union[None, int] = None,
            type_graph_label: str = "categorical",
            final_pool: str = "mean",  # mean, max, sum
            **kwargs
    ):
        super().__init__(dim_features=dim_features, type_graph_label=type_graph_label, **kwargs)
        self.dim_latent = dim_node_embedding[-1]
        self.dim_graph_label = dim_graph_label
        self.dim_local_label = dim_local_label
        self.encode_features = encode_features
        self.type_graph_label = type_graph_label
        self.multi_task = False

        # add feature embedding on node-level
        if self.encode_features:
            self.feature_embedding = ModuleMLP(activation=activation, in_channels=dim_features,
                                               units=dim_node_embedding)
            in_channels = self.dim_latent
        else:
            in_channels = dim_features
        # node-wise GCN pass
        self.node_embedding = ModuleGAT(activation=activation, in_channels=in_channels, units=dim_node_embedding)
        
        # graph-level supervision
        # allow for multi-task learning
        if isinstance(dim_graph_label, list):
            self.multi_task = True
            self.graph_clf = torch.nn.ModuleList([ModuleMLP(activation=activation, in_channels=self.dim_latent,
                                                            units=dim_mlp + [dim_graph_label[i]], 
                                                            type_graph_label=type_graph_label[i],
                                                            prediction_transformation=True) 
                                                            for i in range(len(dim_graph_label))])
        else:
            self.graph_clf = ModuleMLP(activation=activation, in_channels=self.dim_latent,
                                    units=dim_mlp + [dim_graph_label],
                                    type_graph_label=type_graph_label,
                                    prediction_transformation=True)
        # node-level supervision
        if self.use_node_supervision:
            self.node_clf = torch.nn.Linear(in_features=self.dim_latent, out_features=dim_node_label)
        # self-supervision
        if self.use_local_supervision:
            self.local_clf = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.dim_latent, out_features=dim_local_label),
                torch.nn.Softmax(dim=1),
            )

        # defining node pooling function
        if final_pool == "mean":
            self.pool_fn = global_mean_pool
        elif final_pool == "max":
            self.pool_fn = global_max_pool
        elif final_pool == "sum":
            self.pool_fn = global_add_pool
        else:
            ValueError("Final pool method must be on of ['max', 'mean', 'sum']")

    def forward(self, data):
        x = {
            BATCH_KEY_NODE_FEATURES: data.x[BATCH_KEY_NODE_FEATURES],
            BATCH_KEY_GRAPH_LABELS: data.y[BATCH_KEY_GRAPH_LABELS],
        }
        if BATCH_KEY_NODE_LABELS in data.y.keys():
            x[BATCH_KEY_NODE_LABELS] = data.x[BATCH_KEY_NODE_LABELS]

        if BATCH_KEY_LOCAL_ASSIGNMENTS in data.x.keys():
            x[BATCH_KEY_LOCAL_ASSIGNMENTS] = data.x[BATCH_KEY_LOCAL_ASSIGNMENTS]
            x[BATCH_KEY_LOCAL_LABELS] = data.y[BATCH_KEY_LOCAL_LABELS]

        if self.encode_features:
            # embed features on node-level
            x[BATCH_KEY_NODE_FEATURE_EMBEDDING] = self.feature_embedding(x[BATCH_KEY_NODE_FEATURES])
            input_key = BATCH_KEY_NODE_FEATURE_EMBEDDING
        else:
            input_key = BATCH_KEY_NODE_FEATURES

        batch = data.batch if "batch" in data.keys else None  # defining nodes batch

        # forward pass of node-wise GCN
        x[BATCH_KEY_NODE_EMBEDDING] = self.node_embedding(x=x[input_key], edge_index=data.edge_index)
        if self.use_node_supervision:
            x[BATCH_KEY_NODE_YHAT] = self.node_clf(x[BATCH_KEY_NODE_EMBEDDING])

        if self.use_local_supervision:
            # define local assignment per batch for each node
            if not batch is None:
                batch_local_assignment = batch * self.dim_groups_local + x[BATCH_KEY_LOCAL_ASSIGNMENTS]
                batch_local_assignment = batch_local_assignment.to(torch.int64)
            else:
                batch_local_assignment = x[BATCH_KEY_LOCAL_ASSIGNMENTS]
            z = global_mean_pool(x[BATCH_KEY_NODE_EMBEDDING], batch=batch_local_assignment)
            x[BATCH_KEY_LOCAL_YHAT] = self.local_clf(z)

        # Node pooling to obtain graph embedding
        x[BATCH_KEY_GRAPH_EMBEDDING] = self.pool_fn(x[BATCH_KEY_NODE_EMBEDDING], batch=batch)
        # reshape graph labels to (n_graphs, dim_graph_label)
        try:
            num_graphs = data._num_graphs
        except:
            num_graphs = 1

        if self.multi_task:
            for i, l in enumerate(x[BATCH_KEY_GRAPH_LABELS].keys()):
                x[BATCH_KEY_GRAPH_LABELS][l] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS][l], (num_graphs, self.dim_graph_label[i]))
        else:
            x[BATCH_KEY_GRAPH_LABELS] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS], (num_graphs, self.dim_graph_label))

        # graph level supervision (FC)
        if self.multi_task:
            x[BATCH_KEY_GRAPH_YHAT] = [clf(x[BATCH_KEY_GRAPH_EMBEDDING]) for clf in self.graph_clf]
        else:
            x[BATCH_KEY_GRAPH_YHAT] = self.graph_clf(x[BATCH_KEY_GRAPH_EMBEDDING])
        return x


class GraphAttentionInteractionEmbedding(SampleEmbeddingBase):

    def __init__(
            self,
            dim_features: int,
            dim_graph_label: int,
            dim_node_embedding: List[int],
            dim_mlp: List[int],
            encode_features: bool = True,
            activation: torch.nn.Module = torch.nn.LeakyReLU(0.1),
            dim_local_label: Union[None, int] = None,
            dim_node_label: Union[None, int] = None,
            type_graph_label: str = "categorical",
            final_pool: str = "mean",  # mean, max, sum
            **kwargs
    ):
        super().__init__(dim_features=dim_features, type_graph_label=type_graph_label, **kwargs)
        self.dim_latent = dim_node_embedding[-1]
        self.dim_graph_label = dim_graph_label
        self.dim_local_label = dim_local_label
        self.encode_features = encode_features
        self.type_graph_label = type_graph_label
        self.multi_task = False

        # add feature embedding on node-level
        if self.encode_features:
            self.feature_embedding = ModuleMLP(activation=activation, in_channels=dim_features,
                                               units=dim_node_embedding)
            in_channels = self.dim_latent
        else:
            in_channels = dim_features
        # node-wise GCN pass
        self.node_embedding = ModuleGATInteraction(activation=activation, in_channels=in_channels, units=dim_node_embedding)

        # graph-level supervision
        # allow for multi-task learning
        if isinstance(dim_graph_label, list):
            self.multi_task = True
            self.graph_clf = torch.nn.ModuleList([ModuleMLP(activation=activation, in_channels=self.dim_latent,
                                                            units=dim_mlp + [dim_graph_label[i]],
                                                            type_graph_label=type_graph_label[i],
                                                            prediction_transformation=True)
                                                  for i in range(len(dim_graph_label))])
        else:
            self.graph_clf = ModuleMLP(activation=activation, in_channels=self.dim_latent,
                                       units=dim_mlp + [dim_graph_label],
                                       type_graph_label=type_graph_label,
                                       prediction_transformation=True)
        # node-level supervision
        if self.use_node_supervision:
            self.node_clf = torch.nn.Linear(in_features=self.dim_latent, out_features=dim_node_label)
        # self-supervision
        if self.use_local_supervision:
            self.local_clf = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.dim_latent, out_features=dim_local_label),
                torch.nn.Softmax(dim=1),
            )

        # defining node pooling function
        if final_pool == "mean":
            self.pool_fn = global_mean_pool
        elif final_pool == "max":
            self.pool_fn = global_max_pool
        elif final_pool == "sum":
            self.pool_fn = global_add_pool
        else:
            ValueError("Final pool method must be on of ['max', 'mean', 'sum']")

    def forward(self, data):
        x = {
            BATCH_KEY_NODE_FEATURES: data.x[BATCH_KEY_NODE_FEATURES],
            BATCH_KEY_GRAPH_LABELS: data.y[BATCH_KEY_GRAPH_LABELS],
        }
        if BATCH_KEY_NODE_LABELS in data.y.keys():
            x[BATCH_KEY_NODE_LABELS] = data.x[BATCH_KEY_NODE_LABELS]

        if BATCH_KEY_LOCAL_ASSIGNMENTS in data.x.keys():
            x[BATCH_KEY_LOCAL_ASSIGNMENTS] = data.x[BATCH_KEY_LOCAL_ASSIGNMENTS]
            x[BATCH_KEY_LOCAL_LABELS] = data.y[BATCH_KEY_LOCAL_LABELS]

        if self.encode_features:
            # embed features on node-level
            x[BATCH_KEY_NODE_FEATURE_EMBEDDING] = self.feature_embedding(x[BATCH_KEY_NODE_FEATURES])
            input_key = BATCH_KEY_NODE_FEATURE_EMBEDDING
        else:
            input_key = BATCH_KEY_NODE_FEATURES

        batch = data.batch if "batch" in data.keys else None  # defining nodes batch

        # forward pass of node-wise GCN
        x[BATCH_KEY_NODE_EMBEDDING] = self.node_embedding(x=x[input_key], edge_index=data.edge_index)
        if self.use_node_supervision:
            x[BATCH_KEY_NODE_YHAT] = self.node_clf(x[BATCH_KEY_NODE_EMBEDDING])

        if self.use_local_supervision:
            # define local assignment per batch for each node
            if not batch is None:
                batch_local_assignment = batch * self.dim_groups_local + x[BATCH_KEY_LOCAL_ASSIGNMENTS]
                batch_local_assignment = batch_local_assignment.to(torch.int64)
            else:
                batch_local_assignment = x[BATCH_KEY_LOCAL_ASSIGNMENTS]
            z = global_mean_pool(x[BATCH_KEY_NODE_EMBEDDING], batch=batch_local_assignment)
            x[BATCH_KEY_LOCAL_YHAT] = self.local_clf(z)

        # Node pooling to obtain graph embedding
        x[BATCH_KEY_GRAPH_EMBEDDING] = self.pool_fn(x[BATCH_KEY_NODE_EMBEDDING], batch=batch)
        # reshape graph labels to (n_graphs, dim_graph_label)
        try:
            num_graphs = data._num_graphs
        except:
            num_graphs = 1

        if self.multi_task:
            for i, l in enumerate(x[BATCH_KEY_GRAPH_LABELS].keys()):
                x[BATCH_KEY_GRAPH_LABELS][l] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS][l],
                                                             (num_graphs, self.dim_graph_label[i]))
        else:
            x[BATCH_KEY_GRAPH_LABELS] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS], (num_graphs, self.dim_graph_label))

        # graph level supervision (FC)
        if self.multi_task:
            x[BATCH_KEY_GRAPH_YHAT] = [clf(x[BATCH_KEY_GRAPH_EMBEDDING]) for clf in self.graph_clf]
        else:
            x[BATCH_KEY_GRAPH_YHAT] = self.graph_clf(x[BATCH_KEY_GRAPH_EMBEDDING])
        return x


class GINEmbedding(SampleEmbeddingBase):
    def __init__(
            self,
            dim_features: int,
            dim_graph_label: List[int],  # to allow for multi-task learning
            dim_node_embedding: List[int],
            dim_mlp: List[int],
            activation: torch.nn.Module = torch.nn.LeakyReLU(0.1),
            dim_local_label: Union[None, int] = None,
            dim_node_label: Union[None, int] = None,
            encode_features: bool = True,
            type_graph_label: str = "categorical",
            final_pool: str = "mean",  # mean, max, sum
            **kwargs
    ):
        super().__init__(dim_features=dim_features, **kwargs)
        self.dim_latent = dim_node_embedding[-1]
        self.dim_graph_label = dim_graph_label
        self.dim_local_label = dim_local_label
        self.multi_task = False
        self.encode_features = encode_features
        self.type_graph_label = type_graph_label

        if self.encode_features:
            self.feature_embedding = ModuleMLP(activation=activation, in_channels=dim_features,
                                               units=dim_node_embedding)
            in_channels = self.dim_latent
        else:
            in_channels = dim_features

        self.node_embedding = ModuleGIN(activation=activation, in_channels=in_channels, units=dim_node_embedding)
        if isinstance(dim_graph_label, list):
            self.multi_task = True
            self.graph_clf = torch.nn.ModuleList([
                ModuleMLP(activation=activation, in_channels=self.dim_latent*len(dim_node_embedding),
                          units=dim_mlp + [dim_graph_label[i]], type_graph_label=self.type_graph_label[i],
                          prediction_transformation=True)
                for i in range(len(dim_graph_label))])
        else:
            self.graph_clf = ModuleMLP(activation=activation, in_channels=self.dim_latent*len(dim_node_embedding),
                                       units=dim_mlp + [dim_graph_label], prediction_transformation=True)

        # node-level supervision
        if self.use_node_supervision:
            self.node_clf = torch.nn.Linear(in_features=self.dim_latent, out_features=dim_node_label)
        # self-supervision
        if self.use_local_supervision:
            self.local_clf = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.dim_latent*len(dim_node_embedding), out_features=dim_local_label),
                torch.nn.Softmax(dim=1),
            )

        # defining node pooling function
        if final_pool == "mean":
            self.pool_fn = global_mean_pool
        elif final_pool == "max":
            self.pool_fn = global_max_pool
        elif final_pool == "sum":
            self.pool_fn = global_add_pool
        else:
            ValueError("Final pool method must be on of ['max', 'mean', 'sum']")

    def forward(self, data):
        x = {
            BATCH_KEY_NODE_FEATURES: data.x[BATCH_KEY_NODE_FEATURES],
            BATCH_KEY_GRAPH_LABELS: data.y[BATCH_KEY_GRAPH_LABELS],
        }

        # spatial-only model does not use node features
        if BATCH_KEY_NODE_LABELS in data.y.keys():
            x[BATCH_KEY_NODE_LABELS] = data.x[BATCH_KEY_NODE_LABELS]

        if BATCH_KEY_LOCAL_ASSIGNMENTS in data.x.keys():
            x[BATCH_KEY_LOCAL_ASSIGNMENTS] = data.x[BATCH_KEY_LOCAL_ASSIGNMENTS]
            x[BATCH_KEY_LOCAL_LABELS] = data.y[BATCH_KEY_LOCAL_LABELS]

        if self.encode_features:
            # embed features on node-level
            x[BATCH_KEY_NODE_FEATURE_EMBEDDING] = self.feature_embedding(x[BATCH_KEY_NODE_FEATURES])
            input_key = BATCH_KEY_NODE_FEATURE_EMBEDDING
        else:
            input_key = BATCH_KEY_NODE_FEATURES

        batch = data.batch if "batch" in data.keys else None # defining nodes batch

        # forward pass of node-wise GIN
        h_ = self.node_embedding(x=x[input_key], edge_index=data.edge_index)
        h = []

        # Node pooling to obtain graph embedding
        for h_i in h_:
            h.append(self.pool_fn(h_i, batch))

        x[BATCH_KEY_GRAPH_EMBEDDING] = torch.cat(h, dim=1)

        # node-level supervision
        if self.use_node_supervision:
            x[BATCH_KEY_NODE_YHAT] = self.node_clf(x[BATCH_KEY_NODE_EMBEDDING])
        
        if self.use_local_supervision:
            # define local assignment per batch for each node
            if not batch is None:
                batch_local_assignment = batch * self.dim_groups_local + x[BATCH_KEY_LOCAL_ASSIGNMENTS]
                batch_local_assignment = batch_local_assignment.to(torch.int64)
            else:
                batch_local_assignment = x[BATCH_KEY_LOCAL_ASSIGNMENTS]
            z = global_mean_pool(x[BATCH_KEY_NODE_EMBEDDING], batch=batch_local_assignment)
            x[BATCH_KEY_LOCAL_YHAT] = self.local_clf(z)

        # reshape graph labels to (n_graphs, dim_graph_label) - multi-task learning
        try:
            num_graphs = data._num_graphs
        except:
            num_graphs = 1
        if isinstance(x[BATCH_KEY_GRAPH_LABELS], dict):
            for i, l in enumerate(x[BATCH_KEY_GRAPH_LABELS].keys()):
                x[BATCH_KEY_GRAPH_LABELS][l] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS][l],
                                                             (num_graphs, self.dim_graph_label[i]))
        else:
            x[BATCH_KEY_GRAPH_LABELS] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS], (num_graphs, self.dim_graph_label))
        # graph level supervision (FC)
        if self.multi_task:
            x[BATCH_KEY_GRAPH_YHAT] = [clf(x[BATCH_KEY_GRAPH_EMBEDDING]) for clf in self.graph_clf]
        else:
            x[BATCH_KEY_GRAPH_YHAT] = self.graph_clf(x[BATCH_KEY_GRAPH_EMBEDDING])
        return x       


class MultiInstanceEmbedding(SampleEmbeddingBase):

    def __init__(
            self,
            dim_features: int,
            dim_graph_label: int,
            dim_node_embedding: List[int],
            dim_mlp: List[int],
            activation: torch.nn.Module = torch.nn.LeakyReLU(0.1),
            dim_local_label: Union[None, int] = None,
            dim_node_label: Union[None, int] = None,
            type_graph_label: str = "categorical",
            final_pool: str = "mean",  # "mean", "max", "sum"
            **kwargs
    ):
        super().__init__(dim_features=dim_features, type_graph_label=type_graph_label, **kwargs)
        self.dim_latent = dim_node_embedding[-1]
        self.dim_graph_label = dim_graph_label
        self.node_embedding = ModuleMLP(activation=activation, in_channels=dim_features, units=dim_node_embedding)
        self.multi_task = False

        if isinstance(dim_graph_label, list):
            self.multi_task = True
            self.graph_clf = torch.nn.ModuleList([ModuleMLP(activation=activation, in_channels=self.dim_latent,
                                                            units=dim_mlp + [dim_graph_label[i]], 
                                                            type_graph_label=type_graph_label[i],
                                                            prediction_transformation=True) 
                                                            for i in range(len(dim_graph_label))])
        else:
            self.graph_clf = ModuleMLP(activation=activation, in_channels=self.dim_latent,
                                    units=dim_mlp + [dim_graph_label],
                                    type_graph_label=type_graph_label,
                                    prediction_transformation=True)
        if self.use_node_supervision:
            self.node_clf = torch.nn.Linear(in_features=self.dim_latent, out_features=dim_node_label)
        
        if final_pool == "mean":
            self.pool_fn = global_mean_pool
        elif final_pool == "max":
            self.pool_fn = global_max_pool
        elif final_pool == "sum":
            self.pool_fn = global_add_pool
        else:
            ValueError("Final pool method must be on of ['max', 'mean', 'sum']")

    def forward(self, data):
        x = {
            BATCH_KEY_NODE_FEATURES: data.x[BATCH_KEY_NODE_FEATURES],
            BATCH_KEY_GRAPH_LABELS: data.y[BATCH_KEY_GRAPH_LABELS],
        }
        if BATCH_KEY_NODE_LABELS in data.y.keys():
            x[BATCH_KEY_NODE_LABELS] = data.x[BATCH_KEY_NODE_LABELS]
        if BATCH_KEY_LOCAL_ASSIGNMENTS in data.y.keys():
            x[BATCH_KEY_LOCAL_ASSIGNMENTS] = data.x[BATCH_KEY_LOCAL_ASSIGNMENTS]
            x[BATCH_KEY_LOCAL_LABELS] = data.y[BATCH_KEY_LOCAL_LABELS]
             
        x[BATCH_KEY_NODE_EMBEDDING] = self.node_embedding(x[BATCH_KEY_NODE_FEATURES])
        if self.use_node_supervision:
            x[BATCH_KEY_NODE_YHAT] = self.node_clf(x[BATCH_KEY_NODE_EMBEDDING])

        batch = data.batch if "batch" in data.keys else None

        x[BATCH_KEY_GRAPH_EMBEDDING] = self.pool_fn(x[BATCH_KEY_NODE_EMBEDDING], batch=batch)
        try:
            num_graphs = data._num_graphs
        except:
            num_graphs = 1

        # reshape graph labels to (n_graphs, dim_graph_label)
        if self.multi_task:
            for i, l in enumerate(x[BATCH_KEY_GRAPH_LABELS].keys()):
                x[BATCH_KEY_GRAPH_LABELS][l] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS][l], (num_graphs, self.dim_graph_label[i]))
        else:
            x[BATCH_KEY_GRAPH_LABELS] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS], (num_graphs, self.dim_graph_label))
        
        # graph-level supervision
        if self.multi_task:
            x[BATCH_KEY_GRAPH_YHAT] = [clf(x[BATCH_KEY_GRAPH_EMBEDDING]) for clf in self.graph_clf]
        else:
            x[BATCH_KEY_GRAPH_YHAT] = self.graph_clf(x[BATCH_KEY_GRAPH_EMBEDDING])

        return x


class MeanEmbedding(SampleEmbeddingBase):
    def __init__(
            self,
            dim_features: int,
            dim_graph_label: Union[int, List[int]],
            dim_mlp: List[int],
            type_graph_label: str = "categorical",
            activation: torch.nn.Module = torch.nn.ReLU(),
            **kwargs
    ):
        print(f"{type_graph_label}")
        super().__init__(dim_features=dim_features, type_graph_label=type_graph_label, **kwargs)
        self.dim_graph_label = dim_graph_label
        self.multi_task = False

        print(f"{dim_graph_label=}")
        if isinstance(dim_graph_label, list):
            self.multi_task = True
            self.clf = torch.nn.ModuleList([ModuleMLP(activation=activation, 
                                                      in_channels=dim_features,
                                                      units=dim_mlp + [dim_graph_label[i]], 
                                                      type_graph_label=type_graph_label[i],
                                                      prediction_transformation=True) 
                                                    for i in range(len(dim_graph_label))])
        else:
            self.clf = ModuleMLP(activation=activation,
                                in_channels=dim_features,
                                units=dim_mlp + [dim_graph_label],
                                type_graph_label=type_graph_label,
                                prediction_transformation=True)

    def forward(self, data):
        # Expects a list of node attributes: features and labels. Graph labels can be a list.
        x = {
            BATCH_KEY_NODE_FEATURES: data.x[BATCH_KEY_NODE_FEATURES],
            BATCH_KEY_GRAPH_LABELS: data.y[BATCH_KEY_GRAPH_LABELS],
        }

        batch = data.batch if "batch" in data.keys else None
        x[BATCH_KEY_NODE_FEATURES] = global_mean_pool(x[BATCH_KEY_NODE_FEATURES], batch=batch)

        # reshape graph labels to (n_graphs, dim_graph_label)
        try:
            num_graphs = data._num_graphs
        except:
            num_graphs = 1
        if self.multi_task:
            for i, l in enumerate(x[BATCH_KEY_GRAPH_LABELS].keys()):
                x[BATCH_KEY_GRAPH_LABELS][l] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS][l], (num_graphs, self.dim_graph_label[i]))
        else:
            x[BATCH_KEY_GRAPH_LABELS] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS], (num_graphs, self.dim_graph_label))
        
        # allow for multi-task learning
        if self.multi_task:
            x[BATCH_KEY_GRAPH_YHAT] = [clf(x[BATCH_KEY_NODE_FEATURES]) for clf in self.clf]
        else:
            x[BATCH_KEY_GRAPH_YHAT] = self.clf(x[BATCH_KEY_NODE_FEATURES])
        return x


class AggregationConditionEmbedding(SampleEmbeddingBase):
    def __init__(
            self,
            dim_features: int,
            dim_graph_label: int,
            dim_node_embedding: List[int],
            dim_mlp: List[int],
            activation: torch.nn.Module = torch.nn.LeakyReLU(0.1),
            dim_local_label: Union[None, int] = None,
            dim_node_label: int = None,
            type_graph_label: str = "categorical",
            final_pool: str = "mean",  # "mean", "max", "sum"
            **kwargs
    ):
        super().__init__(dim_features=dim_features, **kwargs)
        self.dim_latent = dim_node_embedding[-1]
        self.dim_graph_label = dim_graph_label
        self.dim_node_label = dim_node_label

        self.node_embedding = ModuleMLP(activation=activation, in_channels=dim_features, units=dim_node_embedding)
        self.graph_clf = ModuleMLP(activation=activation, in_channels=self.dim_latent,
                                   units=dim_mlp + [dim_graph_label],
                                   type_graph_label=type_graph_label,
                                   prediction_transformation=True)
        
        if self.use_node_supervision:
            self.node_clf = torch.nn.Linear(in_features=self.dim_latent, out_features=dim_node_label)
        if self.use_local_supervision:
            self.local_clf = torch.nn.Linear(in_features=self.dim_latent, out_features=dim_local_label)
        
        if final_pool == "mean":
            self.pool_fn = global_mean_pool
        elif final_pool == "max":
            self.pool_fn = global_max_pool
        elif final_pool == "sum":
            self.pool_fn = global_add_pool
        else:
            ValueError("Final pool method must be on of ['max', 'mean', 'sum']")

    def forward(self, data):
        # Expects a list of node attributes: features and labels. Graph labels can be a list.
        x = {
            BATCH_KEY_NODE_FEATURES: data.x[BATCH_KEY_NODE_FEATURES],
            BATCH_KEY_NODE_LABELS: data.x[BATCH_KEY_NODE_LABELS], 
            BATCH_KEY_GRAPH_LABELS: data.y[BATCH_KEY_GRAPH_LABELS],
        }
        x[BATCH_KEY_NODE_LABELS] = torch.argmax(x[BATCH_KEY_NODE_LABELS], dim=1)
        
        batch = data.batch if "batch" in data.keys else None

        if batch is None:
            batch_node_label = x[BATCH_KEY_NODE_LABELS]
        else:
            batch_node_label = batch * self.dim_node_label + x[BATCH_KEY_NODE_LABELS]
            batch_node_label = batch_node_label.to(torch.int64)

        batch_size = torch.max(batch)
        size = batch_size*self.dim_node_label + self.dim_node_label
        
        x[BATCH_KEY_NODE_FEATURES] = global_mean_pool(x[BATCH_KEY_NODE_FEATURES], batch=batch_node_label, size=size)
        x[BATCH_KEY_NODE_EMBEDDING] = self.node_embedding(x[BATCH_KEY_NODE_FEATURES])
        
        device = batch.device
        # reduce batches to aggregate node embeddings
        batch_agg = torch.tensor([], dtype=torch.int64).to(device)
        for b in torch.unique(batch):
            batch_agg = torch.cat((batch_agg, torch.full((self.dim_node_label,), b, dtype=torch.int64, device=device)))

        x[BATCH_KEY_GRAPH_EMBEDDING] = self.pool_fn(x[BATCH_KEY_NODE_EMBEDDING], batch=batch_agg)

        # reshape graph labels to (n_graphs, dim_graph_label)
        try:
            num_graphs = data._num_graphs
        except:
            num_graphs = 1
        x[BATCH_KEY_GRAPH_LABELS] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS], (num_graphs, self.dim_graph_label))
        x[BATCH_KEY_GRAPH_YHAT] = self.graph_clf(x[BATCH_KEY_GRAPH_EMBEDDING])
        return x


class DispersionEmbedding(SampleEmbeddingBase):
    # multi-layer perceptron based on average graph niches features.
    # used for cell type models only
    def __init__(self):
        return
    
    def forwad(self):
        return
    
    
class GraphSpatialOnlyEmbedding(SampleEmbeddingBase):
    def __init__(
            self,
            dim_features: int,
            dim_graph_label: int,
            dim_node_embedding: List[int],
            dim_mlp: List[int],
            activation: torch.nn.Module = torch.nn.LeakyReLU(0.1),
            dim_local_label: Union[None, int] = None,
            type_graph_label: str = "categorical",
            final_pool: str = "mean",  # mean, max, sum
            **kwargs
    ):
        super().__init__(dim_features=dim_features, **kwargs)
        self.dim_latent = dim_node_embedding[-1]
        self.dim_graph_label = dim_graph_label
        self.dim_local_label = dim_local_label
        self.dim_features = dim_features
        # node-wise GCN pass
        self.node_embedding = ModuleGCN(activation=activation, in_channels=dim_features, units=dim_node_embedding)
        # graph-level supervision
        self.graph_clf = ModuleMLP(activation=activation, in_channels=self.dim_latent,
                                   units=dim_mlp + [dim_graph_label],
                                   type_graph_label=type_graph_label,
                                   prediction_transformation=True)
        
        # defining node pooling function
        if final_pool == "mean":
            self.pool_fn = global_mean_pool
        elif final_pool == "max":
            self.pool_fn = global_max_pool
        elif final_pool == "sum":
            self.pool_fn = global_add_pool
        else:
            ValueError("Final pool method must be on of ['max', 'mean', 'sum']")

    def forward(self, data):
        x = {
            BATCH_KEY_NODE_FEATURES: data.x[BATCH_KEY_NODE_FEATURES],
            BATCH_KEY_GRAPH_LABELS: data.y[BATCH_KEY_GRAPH_LABELS],
        }

        batch = data.batch if "batch" in data.keys else None # defining nodes batch

        device = batch.device
        # set all node features to ones
        x[BATCH_KEY_NODE_FEATURES] = torch.ones((x[BATCH_KEY_NODE_FEATURES].shape[0], self.dim_features), device=device)

        # forward pass of node-wise GCN
        x[BATCH_KEY_NODE_EMBEDDING] = self.node_embedding(x=x[BATCH_KEY_NODE_FEATURES], edge_index=data.edge_index)
        
        # Node pooling to obtain graph embedding
        x[BATCH_KEY_GRAPH_EMBEDDING] = self.pool_fn(x[BATCH_KEY_NODE_EMBEDDING], batch=batch)

        # reshape graph labels to (n_graphs, dim_graph_label)
        try:
            num_graphs = data._num_graphs
        except:
            num_graphs = 1
        x[BATCH_KEY_GRAPH_LABELS] = torch.reshape(x[BATCH_KEY_GRAPH_LABELS], (num_graphs, self.dim_graph_label))
        
        # graph level supervision (FC)
        x[BATCH_KEY_GRAPH_YHAT] = self.graph_clf(x[BATCH_KEY_GRAPH_EMBEDDING])
        return x    
