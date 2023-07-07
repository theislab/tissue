import torch.nn as nn


class ModuleMLP(nn.Sequential):
    def __init__(self, in_channels, units, activation: nn.Module, prediction_transformation: bool = False,
                 type_graph_label: str = "categorical"):
        layers = []
        prev_dim = in_channels
        for i, dim in enumerate(units):
            is_last = i == len(units) - 1
            layers += [nn.Linear(prev_dim, dim)]
            if not is_last:
                if activation:
                    layers += [activation]
            prev_dim = dim
        # Task-specific activation:
        if prediction_transformation:
            task_activation = None
            if type_graph_label == "percentage":
                task_activation = nn.Sigmoid()
            elif type_graph_label == "categorical":
                task_activation = None  # CCE expects logits
            elif type_graph_label == "continuous":
                task_activation = None  # output over all real numbers
            elif type_graph_label == "survival":
                task_activation = nn.ReLU(0.1)
            # apply activation
            if task_activation:
                layers += [task_activation]

        super().__init__(*layers)
