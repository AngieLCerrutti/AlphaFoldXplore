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
### Functions

```python
set_up()
#Downloads and installs AlphaFold.

predict(dir_string)
#Does not work without running set_up() beforehand.
#Predicts the terciary structure of a single protein (or a group of them) by reading a FASTA file.
#Simplified to default parameters to speed up the process. 
#Returns a dictionary with the predicted results inside in form of objects. (how to use them is pending)

extract_zips(dir_string)
#Unneeded to use normally. 
#Creates two folders 'json_files' and 'pdb_files' and stores the .json and .pdb files from the folders inside.

get_pae_files(dir_string)
#Reads all .json files from a folder (by default, 'json_files'). Assumes the .json files are those from the predictions.
#Returns a dictionary.

get_plddt_files(dir_string)
#Reads all .pdb files from a folder (by default, 'pdb_files') and extracts the pLDDT values from its CA atoms.
#Returns a dictionary.

pae_results(protein_1,protein_2)
#Compares the PAE values of two proteins by reading .json files and creating heatmaps. Protein_2 is optional.
#Admits both strings and entries from the get_pae_files() dictionary.
#Prints heatmaps.
```
![pae](https://user-images.githubusercontent.com/62774640/174699169-3e1f19b3-2ac4-41db-afed-71db8fd18c79.jpeg)

```python
plddt_results(protein_1,protein_2)
#Compares the pLDDT values of two proteins by reading .pdb files and plotting values of all CA atoms. Protein_2 is optional.
#Admits both strings and entries from the get_plddt_files() dictionary.
#Prints a plot.
```
![plddt](https://user-images.githubusercontent.com/62774640/174700466-921443d1-bee0-4a91-aa85-98b66b558242.jpeg)


```python
superimpose_proteins(protein_1,protein_2)
#Rotates and translates the molecular structure of protein_2 to superimpose (match) it with protein_1.
#Both proteins must be of the same length for it to work properly.
#Creates a .pdb file on the root folder named "superimposed_{filename}.pdb".
#Admits strings.
#Prints the mean RMSD.
#Returns the name of the new PDB file created.

calc_individual_rmsd(protein_1,protein_2,start,end)
#Calculates the individual RMSD (Root mean-square deviation) between CA atoms of both proteins.
#protein_1 and protein_2 must be strings. start and end must be positive int numbers and are optional.
#Plots the result and prints the mean RMSD.
#Returns a list with the values per CA pair.
```

```python
#As for the predicted methods object, its methods are:

plot_pae(optional_object_for_comparison)

plot_plddt(optional_object_for_comparison)

fit(sample_object)

rmsd(sample_object,start,end)

view(optional_object_for_comparison)
```
![rmsd](https://user-images.githubusercontent.com/62774640/174699787-d526c0d6-26d7-4ec4-93a2-d0762e1af301.jpeg)

## Documentation

AlphaFoldXplore can be ran infinitely, or rather, as long as Collab allows you to do so uninterrupted. Hence the usefulness in dropping a long multiFASTA file for it to do its work.

Please note that AlphaFoldXplore will only predict properly sequences of up to 650 residues on a free Colab machine. Using longer sequences with those resources is very likely to crash your session (and for now, lose all your results).

For more details, see the [AlphaFold documentation](https://github.com/deepmind/alphafold).

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

