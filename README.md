# AlphaFoldXplore

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AngieLCerrutti/AlphaFoldXplore/blob/main/example/AlphaFoldXplore_example.ipynb)
Alphafold is a software that has the ability to predict protein structures with an almost experimental accuracy in most cases, but it requires a lot of computational resources such as GPU and CPU, for this reason adaptations were made and the software was able to run optimizing the resources.
How did we do this? We took out the graphical interface and put the default commands, then the code is basically executed and separately the functions were created so that the user can plot what he wants (protein drawing, pLDDT and pae). After that we also created another function that tells you how much deviation there is between two proteins (RMSD). Finally, when you enter a compressed multifasta file it is optimized to run automatically.

![prot](https://user-images.githubusercontent.com/62774640/174698354-a814f773-cd13-4d71-9192-04147fd29b64.jpeg)

## Uso

Install alphafoldxplore.py on a folder without anything else.

## API
```python
import alphafoldxplore
```
### Functions

```python
set_up()
#Downloads and installs AlphaFold.

predict(dir_string)
#Does not work without running set_up() beforehand.
#Predicts the terciary structure of a single protein (or a group of them) by reading a FASTA file.
#Simplified to default parameters to speed up the process. Creates two folders 'json_files' and 'pdb_files' with the results inside.

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

```
plddt_results(protein_1,protein_2)
#Compares the pLDDT values of two proteins by reading .pdb files and plotting values of all CA atoms. Protein_2 is optional.
#Admits both strings and entries from the get_plddt_files() dictionary.
#Prints a plot.
```
![plddt](https://user-images.githubusercontent.com/62774640/174700466-921443d1-bee0-4a91-aa85-98b66b558242.jpeg)


```
watch_proteins(protein_1, protein_2)
#Doesn't work on Colab.
#Creates a widget and allows the visualization of protein structures by reading .pdb files. Protein_2 is optional.
#Admits strings.
#Returns a NGLView view.

superimpose_proteins(protein_1,protein_2)
#Rotates and translates the molecular structure of protein_2 to superimpose (match) it with protein_1.
#Both proteins must be of the same length for it to work properly.
#Creates a .pdb file on the root folder named "superimposed_{filename}.pdb".
#Admits strings.
#Prints the mean RMSD.

calc_individual_rmsd(protein_1,protein_2,start,end)
#Calculates the individual RMSD (Root mean-square deviation) between CA atoms of both proteins.
#protein_1 and protein_2 must be strings. start and end must be positive int numbers and are optional.
#Plots the result and prints the mean RMSD.
#Returns a list with the values per CA pair.
```
![rsmd](https://user-images.githubusercontent.com/62774640/174699787-d526c0d6-26d7-4ec4-93a2-d0762e1af301.jpeg)

## Documentation
For more details, see the [AlphaFold documentation](https://github.com/deepmind/alphafold).

If you have a question, please open an issue.

## Authors

- Elmer Andrés Fernández - Original Idea - [Profile](https://www.researchgate.net/profile/Elmer-Fernandez-2) - [CIDIE]- [CONICET](https://www.conicet.gov.ar) - [UCC](http://www.ucc.edu.ar).
- Juan Ignacio Folco - Bioinformátic - Universidad Católica de Córdoba.
- Angie Lucia Cerrutti -  Bioinformátic - Universidad Católica de Córdoba.
- Veronica Baronetto - Advice - Universidad Católica de Córdoba.
- Pablo Pastore - Advice - Universidad Católica de Córdoba.


## Contributing

To *contribute*, do the following:

Open an issue to discuss possible changes or ask questions
Create a *fork* of the project
Create a new *branch* and push your changes
Create a *pull request*

