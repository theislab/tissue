import numpy as np
import sys
import time
import torch 
import pickle


# manual inputs
data_path_base = sys.argv[1]
data_set = sys.argv[2].lower() # schuerch, metabric, jackson
model_class = sys.argv[3].lower() # GCN, GCNSS, MI, MLP, AggCond, GIN
feature_space = sys.argv[4].lower() # 'molecular', 'celltype'
print(feature_space)
radius_keys = sys.argv[5]
batch_size_keys = sys.argv[6] 
cell_type_coarseness = sys.argv[7] # binary or fine annotation
lr_keys = sys.argv[8]
l2_keys = sys.argv[9]
n_clusters_keys = sys.argv[10]
depth_node_embedding_keys = sys.argv[11]
width_node_embedding_keys = sys.argv[12]
depth_graph_embedding_keys = sys.argv[13]
width_graph_embedding_keys = sys.argv[14]
target_label = sys.argv[15].lower()
multitask_setting = sys.argv[16].lower()    # either 'target', 'small' or 'large'
self_supervision_mode = sys.argv[17].lower()  # 'none', 'ss'
node_supervision_mode = sys.argv[18].lower() # 'none', 'node'
final_pooling = sys.argv[19].lower()  # 'mean', 'max', 'sum'
gs_id = sys.argv[20].lower()
out_path = sys.argv[21]

dispersion = False
spatial = False

try:
    extra_param = sys.argv[22].lower()
    if extra_param.lower() == "disp":
        dispersion = True
    elif extra_param.lower() == "spatial":
        spatial = True
except:
    dispersion = False


print("hyperparameters were: ")
print("1 data_set %s" % data_set)
print("3 model_class %s" % model_class)
print("4 feature_space %s" % feature_space)
print("5 radius_key %s" % radius_keys)
print("6 batch_size_key %s" % batch_size_keys)
print("7 cell_type_coarseness %s" % cell_type_coarseness)
print("8 lr_keys %s" % lr_keys)
print("9 l2_keys %s" % l2_keys)
print("10 n_clusters_key %s" % n_clusters_keys)
print("11 depth_node_embedding %s" % depth_node_embedding_keys)
print("12 width_node_embedding %s" % width_node_embedding_keys)
print("13 width_graph_embedding %s" % depth_graph_embedding_keys)
print("14 width_graph_embedding %s" % width_graph_embedding_keys)
print("15 target_label %s" % target_label)
print("16 multitask_setting %s" % multitask_setting)
print("17 self_supervision_mode %s" % self_supervision_mode)
print("18 node_supervision_mode %s" % node_supervision_mode)
print("19 final_pooling %s" % final_pooling)
print("20 gs_id %s" % gs_id)
print("21 out_path %s" % out_path)
print("22 dispersion %s" % dispersion)
print("23 spatial only %s" % spatial)


if feature_space == 'molecular':
    key_x = "X"
    encode_features = True
elif feature_space == 'celltype':
    key_x = "obsm/node_types"
    encode_features = False
else:
    raise ValueError('fetaure_space not in ["molecular", "celltype"]')

KEY_LOCAL_ASSIGNMENT = None
KEY_LOCAL_LABEL = None
KEY_NODE_LABEL = None
DIM_NODE_LABEL = None

if self_supervision_mode == 'ss':
    self_supervision_label = ['relative_cell_types']
    self_supervision = True
    KEY_LOCAL_ASSIGNMENT = "obsm/local_assignment"
    KEY_LOCAL_LABEL = "uns/graph_covariates/label_tensors/relative_cell_types"
    KEY_NODE_LABEL = "obsm/node_types"
else:
    self_supervision_label = []
    self_supervision = False

if node_supervision_mode == 'node':
    KEY_NODE_LABEL = "obsm/node_types"


if data_set == 'jackson':
    data_path = data_path_base + '/jackson/'
    buffered_data_path = data_path + '/buffer/'
    label_selection_target = ['grade']
    labels_1 = [
        'DFSmonth',
        'tumor_size',
        'ERStatus',
        'PRStatus',
        'HER2Status'
    ]
    label_target_type = ["categorical"]
    labels_1_types = [
        "survival",
        "continuous",
        "categorical",
        "categorical",
        "categorical",
    ]

    label_target_dim = [3] # number of grades
    labels_1_dim = [
        2,
        1,
        2,
        2,
        2,
    ]

    radius_dict = {
        "1": 10,
        "2": 20,
        "3": 50
    }
    cell_type_coarseness = cell_type_coarseness
    DIM_GRAPH_LABELS = 3 # number of grades

elif data_set == 'metabric':
    data_path = data_path_base + '/metabric/'
    buffered_data_path = data_path + '/buffer/'
    label_selection_target = ['grade']
    labels_1 = [
        "tumor_size",
        "ERstatus",
    ]
    label_target_type = ["categorical"]
    labels_1_types = [
        "continuous",
        "categorical",
    ]
    label_target_dim = [3] # number of grades
    labels_1_dim = [
        1,
        2,
    ]
    radius_dict = {
        "1": 10,
        "2": 20,
        "3": 55
    }
    cell_type_coarseness = cell_type_coarseness
    DIM_GRAPH_LABELS = 3 # number of grades
    
elif data_set == "schuerch":
    data_path = data_path_base + '/schuerch/'
    buffered_data_path = data_path + '/buffer/'

    label_selection_target = ['Group']
    label_target_type = ["categorical"]
    labels_1 = [
        'DFS',
        'Diffuse',
        'Klintrup_Makinen',  # slightly finer than Group
        'Sex',
        'Age',
    ]
    labels_1_types = [
        "survival",
        "percentage",
        "categorical",
        "categorical",
        "continuous",
    ]

    label_target_dim = [2] # number of groups
    labels_1_dim = [
        2,
        1,
        3,
        2,
        1,
    ]

    radius_dict = {     # avg node degree:
        "1": 25,    # 2.6
        "2": 50,    # 8.2
        "3": 120    # 40.3
    }
    cell_type_coarseness = cell_type_coarseness
    DIM_GRAPH_LABELS = 2 # number of groups

elif data_set == "lung":
    data_path = data_path_base + '/lung/'
    buffered_data_path = data_path + '/buffer/'

    label_selection_target = ['Stage'] # (I-II: 0, III-IV:1)
    label_target_type = ["categorical"]
    labels_1 = [
        'Smoking Status', # (Smoker: 0, Non-smoker:1)
        'Progression', # (No: 0, Yes: 1)
        'BMI', # (<30: 0, ≥30: 1)
        'Sex', # (Male: 0, Female: 1)
        'Age', # (<75: 0, ≥75: 1)
    ]
    labels_1_types = [
        "categorical",
        "categorical",
        "categorical",
        "categorical",
        "categorical",
    ]

    radius_dict = {     # avg node degree:
        "1": 15,    # 3.7
        "2": 25,    # 10.3
        "3": 55,    # 39.7
    }
    cell_type_coarseness = cell_type_coarseness
    DIM_GRAPH_LABELS = 2 # number of stages    

else:
    raise ValueError('data_origin not recognized')


# Multi task settings
TARGET_KEY = label_selection_target[0]
TARGET_KEY_TYPE = label_target_type[0]

label_selection = label_selection_target
label_selection_type = label_target_type
if multitask_setting == 'target':
    pass
elif multitask_setting == 'multitask':
    label_selection += labels_1
    label_selection_type += labels_1_types
    DIM_GRAPH_LABELS = label_target_dim + labels_1_dim

else:
    raise ValueError("multitask setting %s not recognized" % multitask_setting)

if 'schuerch' in data_set:
    monitor_partition = "train"
    monitor_metric = "loss"
    monitor = f"{monitor_partition}_{monitor_metric}"
    early_stopping = False

else:
    monitor_partition = "val"
    monitor_metric = "loss"
    monitor = f"{monitor_partition}_{monitor_metric}"
    early_stopping = True

if monitor_partition == "val":
    validation_split = 0.1

else:
    validation_split = 0.

test_split = 0.1

if early_stopping:
    epochs = 2000
else:
    epochs = 100

epochs = epochs if "test" not in gs_id else 10  # short run if GS is labeled test

# model and training
# ncv = 6
ncv_test = 3
ncv_val = 3

lr_dict = {
    "1": 0.05,
    "2": 0.005,
    "3": 0.0005,
    "4": 0.00005
}

l2_dict = {
    "1": 0.,
    "2": 1e-6,
    "3": 1e-3,
    "4": 1e0
}

depth_dict = { # for graph embeddings
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 5,
    "5": 10,
    "6": 15,
}

width_dict = { # for both node and graph embeddings
    "0": 4,
    "1": 8,
    "2": 16,
    "3": 32,
    "4": 64
}

depth_node_embedding_dict = { # for node embedding
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3
}

# batch suize
bs_dict = {
    "0": 8,
    "1": 16,
    "2": 32,
    "3": 64,
    "4": 62, # for schuerch
}

# number of clusters
nc_dict = {
    "1": 5,
    "2": 10,
    "3": 20
}

# entropy weight - not used
ew_dict = {
    "1": 1e-4,
    "2": 1e-2,
    "3": 1.,
    "4": 1e2
}

max_steps_per_epoch = 20
max_validation_steps = 10


KEY_GRAPH_LABEL = []
if len(label_selection) > 1:
    for key in label_selection:
        KEY_GRAPH_LABEL.append(f"uns/graph_covariates/label_tensors/{key}")
    TYPE_GRAPH_LABEL = label_selection_type

else:
    KEY_GRAPH_LABEL = f"uns/graph_covariates/label_tensors/{TARGET_KEY}"
    TYPE_GRAPH_LABEL = TARGET_KEY_TYPE


# number of workers
num_workers = 4

# Grid serach sub grid here so that more grid search points can be handled in a single job:
for lr_key in lr_keys.split("+"):
    for l2_key in l2_keys.split("+"):
        for depth_node_embedding_key in depth_node_embedding_keys.split("+"):
            for width_node_embedding_key in width_node_embedding_keys.split("+"):
                for depth_graph_embedding_key in depth_graph_embedding_keys.split("+"):
                    for width_graph_embedding_key in width_graph_embedding_keys.split("+"):
                        for radius in radius_keys.split("+"):
                            for batch_size in batch_size_keys.split("+"):
                                for n_cluster in n_clusters_keys.split("+"):
                                    # Set ID of output
                                    model_id = model_class +  "_" + \
                                        data_set + "_" + \
                                        target_label + \
                                        "_lr" + str(lr_key) + \
                                        "_l2" + str(l2_key) + \
                                        "_den" + str(depth_node_embedding_key) + \
                                        "_win" + str(width_node_embedding_key) + \
                                        "_deg" + str(depth_graph_embedding_key) + \
                                        "_wig" + str(width_graph_embedding_key) + \
                                        "_bs" + str(batch_size) + \
                                        "_r" + str(radius) + \
                                        "_fs" + str(feature_space) + \
                                        "_fp" + str(final_pooling) + \
                                        "_mt" + str(multitask_setting) + \
                                        "_nc" + str(n_cluster) + \
                                        '_ss' + str(self_supervision_mode) + \
                                        '_ns' + str(node_supervision_mode)
                                    
                                    dim_mlp = []
                                    for _ in np.arange(depth_dict[depth_graph_embedding_key]):
                                        dim_mlp.append(width_dict[width_graph_embedding_key])
                                    
                                    dim_node_embedding = []
                                    for _ in np.arange(depth_node_embedding_dict[depth_node_embedding_key]):
                                        dim_node_embedding.append(width_dict[width_node_embedding_key])

                                    print(f"{dim_node_embedding=}")
                                    print(f"{dim_mlp=}")
                                    

                                    spatial = False # boolean to reduce feature size to 1

                                    if "gcn" in model_class:
                                        from tissue.models.graph_classification import GraphEmbedding as Model
                                    elif "mi" in model_class:
                                        from tissue.models.graph_classification import MultiInstanceEmbedding as Model
                                    elif model_class == "reg":
                                        from tissue.models.graph_classification import MeanEmbedding as Model
                                    elif model_class == "gin":
                                        from tissue.models.graph_classification import GINEmbedding as Model
                                    elif model_class == "gat":
                                        from tissue.models.graph_classification import GraphAttentionEmbedding as Model
                                    elif model_class == "gatinteraction":
                                        from tissue.models.graph_classification import GraphAttentionInteractionEmbedding as Model
                                    elif model_class == "aggcond":
                                        from tissue.models.graph_classification import AggregationConditionEmbedding as Model
                                    elif model_class == "spatial":
                                        from tissue.models.graph_classification import GraphSpatialOnlyEmbedding as Model
                                        spatial = True
                                    # TODO: add despersion
                                    else:
                                        raise ValueError("model class %s not recognized" % model_class)
                                   
                                    run_params = {
                                        'model_id': model_id,
                                        'model_class': model_class,
                                        'gs_id': gs_id,
                                        'data_set': data_set,
                                        'radius': radius_dict[radius],
                                        'target_label': target_label,
                                        'graph_label_selection': label_selection,
                                        'featur_space': feature_space,
                                        'learning_rate': lr_dict[lr_key],
                                        'depth_node_embedding': depth_node_embedding_dict[depth_node_embedding_key],
                                        'width_node_embedding': width_dict[width_node_embedding_key],
                                        'depth_graph_embedding': depth_dict[depth_graph_embedding_keys],
                                        'width_graph_embedding': width_dict[width_graph_embedding_key],
                                        'l2_reg': l2_dict[l2_key],
                                        'batch_size': bs_dict[batch_size],
                                        'final_pooling': final_pooling,
                                        'multitask_setting': multitask_setting,
                                        'n_clusters': nc_dict[n_cluster],
                                        'self_supervision_mode': self_supervision_mode,
                                        'node_supervision_mode': node_supervision_mode,
                                    }

                                    fn_out = out_path + "/results/" + model_id
                                    with open(fn_out + '_runparams.pickle', 'wb') as f:
                                        pickle.dump(obj=run_params, file=f)

                                    for i in range(ncv_test):
                                        for j in range(ncv_val):
                                            print("test cv %i" % i)
                                            print("val cv %i" % j)
                                            model_id_cv = model_id + "_cv" + str(i) + "_" + str(j)
                                            fn_out = out_path + "/results/" + model_id_cv

                                             # intialise datamodule
                                            from tissue.data.loading import get_datamodule_from_curated

                                            print(f"{fn_out=}")
                                            print(f"{KEY_LOCAL_ASSIGNMENT=}")
                                            print(f"{KEY_LOCAL_LABEL=}")
                                            print(f"{TYPE_GRAPH_LABEL=}")

                                            data_params = {
                                                'radius': radius_dict[radius],
                                                'key_x': key_x,
                                                'key_graph_supervision': KEY_GRAPH_LABEL,
                                                'type_graph_label': TYPE_GRAPH_LABEL,
                                                'batch_size': bs_dict[batch_size],
                                                'cell_type_coarseness': cell_type_coarseness,
                                                'edge_index': True,
                                                'key_local_assignment': KEY_LOCAL_ASSIGNMENT,
                                                'key_local_supervision': KEY_LOCAL_LABEL,
                                                'key_node_supervision': KEY_NODE_LABEL,
                                                'num_workers': num_workers,
                                                'preprocess': None,
                                                'val_split': validation_split,
                                                'test_split': test_split,
                                                'seed_test': i*10 + i,
                                                'seed_val': j*10 + j,
                                            }

                                            with open(fn_out + '_dataparams.pickle', 'wb') as f:
                                                pickle.dump(obj=data_params, file=f)

                                            datamodule = get_datamodule_from_curated(
                                                    dataset=data_set,
                                                    data_path=data_path,
                                                    buffered_data_path=f"{data_path}/buffer/",
                                                    radius=radius_dict[radius],
                                                    key_x=key_x,
                                                    key_graph_supervision=KEY_GRAPH_LABEL,
                                                    type_graph_label=TYPE_GRAPH_LABEL,
                                                    batch_size=bs_dict[batch_size],
                                                    cell_type_coarseness=cell_type_coarseness,
                                                    edge_index=True,
                                                    key_local_assignment = KEY_LOCAL_ASSIGNMENT,
                                                    key_local_supervision = KEY_LOCAL_LABEL,
                                                    key_node_supervision = KEY_NODE_LABEL,
                                                    num_workers = num_workers,
                                                    preprocess=None,
                                                    val_split = validation_split,
                                                    test_split = test_split,
                                                    seed_test = i*10 + i,
                                                    seed_val = j*10 + j,
                                                )

                                            if self_supervision_mode == "ss":
                                                DIM_GROUPS_LOCAL = nc_dict[n_cluster]
                                                DIM_LOCAL_LABEL = datamodule.dim_node_label
                                                print(f"{DIM_GROUPS_LOCAL=}")
                                                print(f"{DIM_LOCAL_LABEL=}")
                                            else:
                                                DIM_GROUPS_LOCAL = None
                                                DIM_LOCAL_LABEL = None

                                            if spatial:
                                                DIM_FEATURES = 1
                                            else:
                                                DIM_FEATURES = datamodule.dim_features

                                            kwargs = {
                                                "dim_features": DIM_FEATURES,
                                                "dim_graph_label": DIM_GRAPH_LABELS,
                                                "dim_groups_local": DIM_GROUPS_LOCAL,
                                                "dim_local_label": DIM_LOCAL_LABEL,
                                                'type_graph_label': TYPE_GRAPH_LABEL,
                                                "dim_mlp": dim_mlp,
                                                "dim_node_embedding": dim_node_embedding,
                                                "dim_node_label": datamodule.dim_node_label,
                                                "encode_features": encode_features,
                                            }

                                            with open(fn_out + '_modelparams.pickle', 'wb') as f:
                                                pickle.dump(obj=kwargs, file=f)

                                            # initialise model
                                            model = Model(**kwargs)

                                            from tissue.train.train_model import TrainModel

                                            # initialise trainer
                                            trainer = TrainModel()

                                            # initialise estimator
                                            trainer.init_estim(model=model, datamodule=datamodule)

                                            # # train estimator
                                            trainer.estimator.train(lr=lr_dict[lr_key],
                                                            l2=l2_dict[l2_key],
                                                            monitor=monitor,
                                                            max_validation_steps=max_validation_steps,
                                                            epochs=epochs,
                                                            detect_anomaly=True)

                                            # saving
                                            trainer.save(fn=fn_out, save_weights=True)
