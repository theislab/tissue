"""
Exact copy from gpu-spatial-graph-pipeline repository (only node-wise funtions deleted out).
[https://github.com/theislab/gpu-spatial-graph-pipeline/tree/76a84638f35a6dbd9b2d02aaff425699f07d5b43/src/gpu_spatial_graph_pipeline/data]
To be replaced by an imports.
"""
import torch
import pytorch_lightning as pl
from typing import Callable, Literal, Optional, Sequence, Union, List
from torch_geometric.loader import NeighborLoader, DataListLoader
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import RandomNodeSplit
from anndata import AnnData
from tissue.consts import BATCH_KEY_NODE_FEATURES, BATCH_KEY_GRAPH_LABELS

VALID_STAGE = {"fit", "test", None}
VALID_SPLIT = {"node", "graph"}

# TODO: Fix dataloader


class GraphAnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datas: Sequence[Data] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        learning_type: Literal["node", "graph"] = "node",
        idx_train: List[int] = None,
        idx_test: List[int] = None,
        idx_val: List[int] = None,
    ):
        """Manages loading and sampling schemes before loading to GPU.

        Args:
            adata (AnnData, optional): _description_. Defaults to None.
            adata2data_fn (Callable[[AnnData], Union[Sequence[Data], Batch]], optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 1.
            num_workers (int, optional): _description_. Defaults to 1.
            learning_type (Literal[&quot;node&quot;, &quot;graph&quot;], optional):
                If graph is selected batch_size means number of graphs and the adata2data_fn
                is expected to give a list of Data.
                If node is selected batch_size means the number of nodes
                and adata2data_fn is
                expected to give a list of Data objects
                with edge_index attribute.

                 Defaults to "nodewise".


        Raises:
            ValueError: _description_
        """
        # TODO: Fill the docstring

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = datas
        self.seed = seed
        if learning_type not in VALID_SPLIT:
            raise ValueError("Learning type must be one of %r." % VALID_SPLIT)
        self.learning_type = learning_type
        self.first_time = True
        self.val_split = val_split
        self.test_split = test_split
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.idx_val = idx_val

        
        # split datasets  - TODO: on patient level? stratified?
        if idx_train is None:
            generator = torch.Generator().manual_seed(self.seed)

            from torch.utils.data import random_split
            train, val, test = random_split(range(len(self.data)), [1 - (self.val_split + self.test_split), self.val_split, self.test_split], 
                                                        generator=generator)
            self.idx_train = train.indices
            self.idx_test = test.indices
            self.idx_val = val.indices

    def _nodewise_setup(self, stage: Optional[str]):

        if self.first_time:
            self.data = Batch.from_data_list(self.data)
            self.transform = RandomNodeSplit(
                split="train_rest",
                num_val=max(int(self.data.num_nodes * 0.1), 1),
                num_test=max(int(self.data.num_nodes * 0.05), 1),
            )

            self.data = self.transform(self.data)
            self.first_time = False

        if stage == "fit" or stage is None:
            self._train_dataloader = self._spatial_node_loader(
                input_nodes=self.data.train_mask, shuffle=True
            )
            self._val_dataloader = self._spatial_node_loader(
                input_nodes=self.data.val_mask,
            )
        if stage == "test" or stage is None:
            self._test_dataloader = self._spatial_node_loader(
                input_nodes=self.data.test_mask,
            )

    def _graphwise_setup(self, stage: Optional[str]):
        
        from torch.utils.data import Subset

        train_dataset = Subset(self.data, self.idx_train)
        validation_dataset = Subset(self.data, self.idx_val)
        test_dataset = Subset(self.data, self.idx_test)

        if stage == "fit" or stage is None:
            self._train_dataloader = self._graph_loader(
                data=train_dataset,
                shuffle=True,
            )
            self._val_dataloader = self._graph_loader(data=validation_dataset)
        if stage == "test" or stage is None:
            self._test_dataloader = self._graph_loader(
                data=test_dataset,
            )

    def setup(self, stage: Optional[str] = None):
        # TODO: Implement each case
        # TODO: Splitting
        # stage = "train" if not stage else stage

        # if stage not in VALID_STAGE:
        #     print(stage)
        #     raise ValueError("Stage must be one of %r." % VALID_STAGE)

        if self.learning_type == "graph":
            if len(self.data) <= 3:
                raise RuntimeError("Not enough graphs in data to do graph-wise learning")
            self._graphwise_setup(stage)

        else:
            self._nodewise_setup(stage)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def predict_dataloader(self, idx: List[int] = None):
        return self._graph_loader(
            data=[self.data[i] for i in idx]
        )

    def _graph_loader(self, data, shuffle=False, **kwargs):
        """Loads from the list of data

        Args:
            shuffle (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        return DataListLoader(
            dataset=data,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs
        )

    def _spatial_node_loader(self, input_nodes, shuffle=False, **kwargs):
        return NeighborLoader(
            self.data,
            num_neighbors=[-1],
            batch_size=self.batch_size,
            input_nodes=input_nodes,
            shuffle=shuffle,
            num_workers=self.num_workers,
            **kwargs
        )
