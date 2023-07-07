import sys
import pickle


# manual inputs
data_path_base = sys.argv[1]
data_set = sys.argv[2].lower() # schuerch, metabric, jackson
model_class = sys.argv[3].lower() # REGDISP, RFDISP, REGSPATIAL, RFSPATIAL
feature_space = sys.argv[4].lower() # 'molecular', 'celltype'
print(feature_space)
radius_keys = sys.argv[5]
cell_type_coarseness = sys.argv[6] # binary or fine annotation
n_clusters_keys = sys.argv[7]
target_label = sys.argv[8].lower()
gs_id = sys.argv[9].lower()
out_path = sys.argv[10]

dispersion_mode = False
node_degree_mode = False

try:
    extra_param = sys.argv[11].lower()
    if extra_param.lower() == "disp":
        dispersion_mode = True
    elif extra_param.lower() == "spatial":
        node_degree_mode = True
except:
    dispersion_mode = False
    node_degree_mode = False


print("hyperparameters were: ")
print("1 data_set %s" % data_set)
print("3 model_class %s" % model_class)
print("4 feature_space %s" % feature_space)
print("5 radius_key %s" % radius_keys)
print("6 cell_type_coarseness %s" % cell_type_coarseness)
print("7 n_clusters_key %s" % n_clusters_keys)
print("8 target_label %s" % target_label)
print("9 gs_id %s" % gs_id)
print("10 out_path %s" % out_path)
print("11 dispersion %s" % dispersion_mode)
print("12 node degree [spatial] %s" % node_degree_mode)


if feature_space == 'molecular':
    key_x = "X"
elif feature_space == 'celltype':
    key_x = "obsm/node_types"
else:
    raise ValueError('fetaure_space not in ["molecular", "celltype"]')

percentile = 80
dispersion_label = "relative_cell_types"


if data_set == 'jackson':
    data_path = data_path_base + '/jackson/'
    buffered_data_path = data_path + '/buffer/'
    TARGET_KEY = 'grade'
    
    radius_dict = {
        "1": 10,
        "2": 20,
        "3": 50
    }
    cell_type_coarseness = cell_type_coarseness

elif data_set == 'metabric':
    data_path = data_path_base + '/metabric/'
    buffered_data_path = data_path + '/buffer/'
    TARGET_KEY = 'grade'

    radius_dict = {
        "1": 10,
        "2": 20,
        "3": 55
    }
    cell_type_coarseness = cell_type_coarseness
    
elif data_set == "schuerch":
    data_path = data_path_base + '/schuerch/'
    buffered_data_path = data_path + '/buffer/'

    TARGET_KEY = 'Group'

    radius_dict = {     # avg node degree:
        "1": 25,    # 2.6
        "2": 50,    # 8.2
        "3": 120    # 40.3
    }
    cell_type_coarseness = cell_type_coarseness

else:
    raise ValueError('data_origin not recognized')


if 'schuerch' in data_set:
    monitor_partition = "train"
    monitor_metric = "loss"
    monitor = f"{monitor_partition}_{monitor_metric}"
    validation_split = 0.

else:
    monitor_partition = "val"
    monitor_metric = "loss"
    monitor = f"{monitor_partition}_{monitor_metric}"
    validation_split = 0.1


test_split = 0.1

# model and training
ncv_test = 3
ncv_val = 3

# number of clusters
nc_dict = {
    "1": 5,
    "2": 10,
    "3": 20
}

print(f"{buffered_data_path=}")
# Grid serach sub grid here so that more grid search points can be handled in a single job:


for radius in radius_keys.split("+"):
        for n_cluster in n_clusters_keys.split("+"):
            # Set ID of output
            model_id = model_class +  "_" + \
                data_set + "_" + \
                target_label + \
                "_r" + str(radius) + \
                "_fs" + str(feature_space) + \
                "_nc" + str(n_cluster) + \
                '_disp' + str(dispersion_mode) + \
                '_node_degree' + str(node_degree_mode)

            
            
            run_params = {
                'model_id': model_id,
                'model_class': model_class,
                'gs_id': gs_id,
                'data_set': data_set,
                'radius': radius_dict[radius],
                'target_label': target_label,
                'featur_space': feature_space,
                'n_clusters': nc_dict[n_cluster],
                'dispersion': dispersion_mode,
                'node_degree': node_degree_mode,
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

                    # intialise data
                    from tissue.data.loading_mean_baselines import get_data_from_curated

                    print(f"{fn_out=}")
                    
                    data = get_data_from_curated(
                            dataset=data_set,
                            data_path=data_path, 
                            radius=radius_dict[radius],

                            node_feature_space=feature_space,
                            key_supervision=TARGET_KEY,
                            
                            buffered_data_path=buffered_data_path,
                            cell_type_coarseness = cell_type_coarseness, 
                            
                            dispersion=dispersion_mode,
                            dispersion_label=dispersion_label,
                            n_cluster=nc_dict[n_cluster],

                            node_degree=node_degree_mode,
                            percentile=percentile,
                            
                            test_split=test_split,
                            val_split=validation_split,

                            seed_test = i*10 + i,
                            seed_val = j*10 + j,
                        )

                    # initialise model
                    if "reg" in model_class:
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(random_state=i*10 + i)
                    elif "rf" in model_class:
                        from sklearn.ensemble import RandomForestClassifier 
                        model = RandomForestClassifier(n_estimators=100, random_state=i*10 + i, warm_start=True)
                    else:
                        raise ValueError("model class %s not recognized" % model_class)

                    
                    # initialise trainer
                    from tissue.train.train_model_baseline import TrainModel
                    trainer = TrainModel()

                    # initialise estimator
                    trainer.init_estim(model=model, data=data, monitor_partition=monitor_partition)

                    # # train estimator
                    trainer.estimator.train()
                    
                    # saving
                    trainer.save(fn=fn_out)


                                            