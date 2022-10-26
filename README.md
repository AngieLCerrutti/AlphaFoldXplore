# AlphaFoldXplore

A Python interface to run [AlphaFold](https://github.com/deepmind/alphafold) to explore, analize and visualize their protein prediction results in a local-like environment. It's designed to minimize effort and resources.

Up to now it's only available on Colab. It allows processing multifasta files holding sequences of length ≤ 600 AminoAcids.
By uploadng a simple or multifasta file to the provided Colab and just pressing the "run" button, it will process all your sequences. Once finished, all the predictions and metrics will be downloaded to your local disk encapsulated into a Python object (please be sure to set up your browser to allow downloads).

## To Run AlphaFoldXplore for prediction please click in the link below
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AngieLCerrutti/AlphaFoldXplore/blob/main/example/AlphaFoldXplore_Colab_SimplePredict.ipynb)

Then you will be able to analize, compare, visualize all your results through AlphaFoldXplore in your local machine with an easy to use API.

### If you whant to export the predicted PDB files or evaluate your results through AlphaFoldXplore, please visit our [Wiki](https://github.com/AngieLCerrutti/AlphaFoldXplore/wiki) files for more examples and further information.

# AlphaFoldXplore Rationale

Alphafold is a software that has the ability to predict protein structures with an almost experimental accuracy in most cases, but it requires a lot of computational resources such as GPU; for this reason, some tweaks were done for the software to be able to run on smaller optimized environments with minimizing user interactions, such as removing any graphical interface and defaulting parameters to the fastest options. 

In addition, AlphaFoldXplore provides several functions through a Object Oriented software API allowing the user plotting the predictions to check prediction quality scores ([pLDDT and PAE metrics](https://www.deepmind.com/publications/enabling-high-accuracy-protein-structure-prediction-at-the-proteome-scale)). In addition we implement some functionalities to explore deviations between two proteins ([RMSD](https://www.sciencedirect.com/science/article/pii/S1359027898000194)), protein overlapping and more. 

![prot](https://user-images.githubusercontent.com/62774640/174698354-a814f773-cd13-4d71-9192-04147fd29b64.jpeg)
<sup>A pair of proteins predicted with AlphaFoldXplore, superimposed against each other and visualized through AlphaFoldXplore.</sup>

## Installation

UNDER CONSTRUCTION

Cloning the Github repo:
```
git clone https://github.com/AngieLCerrutti/AlphaFoldXplore
```
Importing the AlphaFoldXplore module into your notebook:
```python
from AlphaFoldXplore import alphafoldxplore
```

## API

### Requisites (not exhaustive)

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

- Elmer Andrés Fernández (PhD) - Original Idea - [Profile](https://www.researchgate.net/profile/Elmer-Fernandez-2) - CIDIE- [CONICET](https://www.conicet.gov.ar) - [UCC](http://www.ucc.edu.ar). Professor of Intelligent Systems @ UCC.
- Juan Ignacio Folco - Developer and Maintainer - Universidad Católica de Córdoba.
- Angie Lucía Cerrutti - Programming - Universidad Católica de Córdoba.
- Verónica Baronetto (PhD student)- Advice - Universidad Católica de Córdoba.
- Pablo Pastore - Advice - Universidad Católica de Córdoba, Asistant Professor of Intelligent Systems @ UCC.


## Contributing

To *contribute*, do the following:
* Contact efernandez at cidie . ucc . edu . ar
* Open an issue to discuss possible changes or ask questions
* Create a *fork* of the project
* Create a new *branch* and push your changes
* Create a *pull request*

