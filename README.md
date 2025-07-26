# Tissue



## Features

tissue is a model repository in a single python package for the manuscript Ali, M., Richter, S., Etrürk, A. , Fischer, D. S. and Theis, F. "Graph neural networks learn emergent tissue properties from spatial molecular profiles."

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16440987.svg)](https://doi.org/10.5281/zenodo.16440987)

![Screenshot 2023-07-31 at 18 49 03](https://github.com/theislab/tissue/assets/9961724/d0e78c7d-ede8-4a05-8254-5ca0060fbd17)
 
![Screenshot 2023-07-31 at 18 49 37](https://github.com/theislab/tissue/assets/9961724/d2fd82ea-e82f-4d94-a003-c3c9e4a2b541)


## Installation

You can install _Tissue_ via :

```console
$ git clone tissue
$ cd tissue
$ pip install -e .
```

## Requirements

You can install the requirements via:

```console
$ pip install -r requirements.txt
```

## Usage

The repository consists of different components

I. Data loading: datasets can be defined under `data/datasets.py` and pytorch geometric dataloaders are adjusted accordingly in `data/loading.py`

II. Models: graph neural networks and baseline models as described in the paper, the following models can be found under `modules/`:
#### Graph Neural Networks (GNNs)
1. Graph convolutional network (GCN)
2. Graph isomorphism network (GIN)
4. Graph attention network (GAT)


**1. Spatial Tissue Architecture**
- Graph-based models (GIN, GCN)


**2. Graph statistics**
- Density-based models

**3. Single Cell**
- Multiple-instance models

**4. Bulk**
- Multilayer Perceptron (MLP)
- Logistic Regression
- Random Forest



The models can be trained using the training scripts provided under `train/`.

III. Summary and evaluation of models: model evaluation and plotting functions are defined in `train/summaries.py`

IV. Model interpretation: interpretation methods on graph and node embedding levels are implemented under `interpretation/`

   
## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Tissue_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/mayarali/tissue/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/mayarali/tissue/blob/main/LICENSE
[contributor guide]: https://github.com/mayarali/tissue/blob/main/CONTRIBUTING.md
[command-line reference]: https://tissue.readthedocs.io/en/latest/usage.html
