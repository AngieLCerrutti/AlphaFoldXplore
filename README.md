# AlphaFoldXplore

A simple program to simplify the use of [AlphaFold](https://github.com/deepmind/alphafold), visualization of protein structures and other metrics for the end user.

Only for Colab as of the current version.

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AngieLCerrutti/AlphaFoldXplore/blob/main/example/AlphaFoldXplore_example.ipynb)

Alphafold is a software that has the ability to predict protein structures with an almost experimental accuracy in most cases, but it requires a lot of computational resources such as GPU; for this reason, some tweaks were done for the software to be able to run optimizing resources, such as removing any graphical interface and defaulting parameters to the fastest options. 

Separately, some functions were made so that the user can plot the predictions and how trustable they are (protein drawing, [pLDDT and PAE metrics](https://www.deepmind.com/publications/enabling-high-accuracy-protein-structure-prediction-at-the-proteome-scale)). After that, another function that estimates how much deviation there is between two proteins ([RMSD](https://www.sciencedirect.com/science/article/pii/S1359027898000194)) was made. Finally, improvements were made to accept the input of a multiFASTA file.

![prot](https://user-images.githubusercontent.com/62774640/174698354-a814f773-cd13-4d71-9192-04147fd29b64.jpeg)
<sup>A pair of proteins predicted with AlphaFoldXplore and superimposed against each other.</sup>

## Installation

For best results, install ```alphafoldxplore.py``` on an empty folder at first.

The easiest way is to clone the Github repo:
```
git clone https://github.com/AngieLCerrutti/AlphaFoldXplore
```
and then import the module with:
```python
from AlphaFoldXplore import alphafoldxplore
```

Otherwise, since the module is nothing but a few _.py_ files, you can download them alone on your folder of choice:
```
wget -O alphafoldxplore https://github.com/AngieLCerrutti/AlphaFoldXplore/blob/main/alphafoldxplore.py
wget -O alphafoldxplore https://github.com/AngieLCerrutti/AlphaFoldXplore/blob/main/prediction_results.py
chmod +x alphafoldxplore
```
## API

### Requisites

* ``` python >= 3.7 ```
* ``` biopython >= 1.79 ```
* ``` nglview >= 3.0.3 ```
* ``` jax >= 0.3.8 ```
* ``` tqdm >= 4.64.0 ```
* ``` numpy ```
* ``` seaborn ```

For notes about how to use AlphaFoldXplore and its functions, see the [wiki](https://github.com/AngieLCerrutti/AlphaFoldXplore/wiki).

For details about how the predictions work, see the [AlphaFold documentation](https://github.com/deepmind/alphafold).

## Authors

- Elmer Andrés Fernández - Original Idea - [Profile](https://www.researchgate.net/profile/Elmer-Fernandez-2) - CIDIE- [CONICET](https://www.conicet.gov.ar) - [UCC](http://www.ucc.edu.ar).
- Juan Ignacio Folco - Programming - Universidad Católica de Córdoba.
- Angie Lucía Cerrutti -  Programming - Universidad Católica de Córdoba.
- Verónica Baronetto - Advice - Universidad Católica de Córdoba.
- Pablo Pastore - Advice - Universidad Católica de Córdoba.


## Contributing

To *contribute*, do the following:

* Open an issue to discuss possible changes or ask questions
* Create a *fork* of the project
* Create a new *branch* and push your changes
* Create a *pull request*

