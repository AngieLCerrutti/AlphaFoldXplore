# AlphaFoldXplore

A python interface to run  [AlphaFold](https://github.com/deepmind/alphafold) and explore, analize and visualize their protein prediction results in a local like environment. It is designed to minimize effort and resources.

Up to now it is only available on Colab. It allows processing multifasta files holding sequences of length \leq 600 Aminoacids.
By uploadng a simple or multifasta file to the colab and just pressing the run button, it will process all your sequences. Once finish all the rpediction and metrics will be doownloaded to you local disk encapsulated into a python object (Please be sure to set up your browser to automatic download).

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AngieLCerrutti/AlphaFoldXplore/blob/main/example/AlphaFoldXplore_Colab_SimplePredict.ipynb)

The you will be able to analize, compare, visualize all your results trhough AlphaFoldXplore in your local machine with an easy to use API.

# AlphaFoldXplore Rationale

Alphafold is a software that has the ability to predict protein structures with an almost experimental accuracy in most cases, but it requires a lot of computational resources such as GPU; for this reason, some tweaks were done for the software to be able to run on smaller optimized environments with minimizing user interactions, such as removing any graphical interface and defaulting parameters to the fastest options. 

In addition, AlphaFoldXplore provides serval functions through a Object Oriented software API allowing  the user ploting the predictions, prediction quality scores [pLDDT and PAE metrics](https://www.deepmind.com/publications/enabling-high-accuracy-protein-structure-prediction-at-the-proteome-scale)). In addition we implement some functionalities to explore deviations between two proteins ([RMSD](https://www.sciencedirect.com/science/article/pii/S1359027898000194)), protein overlapping and more. 

![prot](https://user-images.githubusercontent.com/62774640/174698354-a814f773-cd13-4d71-9192-04147fd29b64.jpeg)
<sup>A pair of proteins predicted with AlphaFoldXplore, superimposed against each other and visualized trhough AlphaFoldXplore.</sup>

## Installation

UNDER CONSTRUCTION

cloning the Github repo:
```
git clone https://github.com/AngieLCerrutti/AlphaFoldXplore
```
importing the AlphaFoldXplore module into your notebook:
```python
from AlphaFoldXplore import alphafoldxplore
```

## API

### Requisites

* ``` python >= 3.7 ```
* ``` biopython >= 1.79 ```
* ``` nglview >= 3.0.3 ```
* ``` jax >= 0.3.8, <= 0.3.15 ```
* ``` tqdm >= 4.64.0 ```
* ``` numpy ```
* ``` seaborn ```

For notes about how to use AlphaFoldXplore and its functions, see the [wiki](https://github.com/AngieLCerrutti/AlphaFoldXplore/wiki).

For details about how the predictions work and what the results mean, see the [AlphaFold documentation](https://github.com/deepmind/alphafold).

## Authors

- Elmer Andrés Fernández - Original Idea - [Profile](https://www.researchgate.net/profile/Elmer-Fernandez-2) - CIDIE- [CONICET](https://www.conicet.gov.ar) - [UCC](http://www.ucc.edu.ar).
- Juan Ignacio Folco - Developer and Maintener - Universidad Católica de Córdoba.
- Angie Lucía Cerrutti -  Programming - Universidad Católica de Córdoba.
- Verónica Baronetto - Advice - Universidad Católica de Córdoba.
- Pablo Pastore - Advice - Universidad Católica de Córdoba.


## Contributing

To *contribute*, do the following:
* contact efernandez at cidie . ucc . edu . ar
* Open an issue to discuss possible changes or ask questions
* Create a *fork* of the project
* Create a new *branch* and push your changes
* Create a *pull request*

