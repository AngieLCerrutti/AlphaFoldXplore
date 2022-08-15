import sys
sys.path.insert(2, 'AlphaFoldXplore')
import alphafoldxplore as afx
import os

class prediction_results:
    def __init__(self, a=None, b=None, c=0, d=None):
      self.name = a
      self.directory = b
      self.time = c
      self.machine_details = d

    def add_name(self, x):
      self.name = x

    def add_dir(self, x):
      self.directory = x
    
    def add_time(self, x):
      self.time = x

    def add_machine_details(self, x):
      self.machine_details = x

    def plot_pae(self, p2=None): #p2 must be another prediction_results object
      afx.clean()
      afx.extract_zip(self.directory)
      pae_file1 = afx.get_pae_files()
      if p2:
        afx.clean()
        afx.extract_zip(p2.directory)
        pae_file2 = afx.get_pae_files()
        afx.pae_results(pae_file1[list(pae_file1)[0]],pae_file2[list(pae_file2)[0]])
      else:
        afx.pae_results(pae_file1[list(pae_file1)[0]])
      afx.clean()

    def plot_plddt(self, p2=None): #p2 must be another prediction_results object
      afx.clean()
      afx.extract_zip(self.directory)
      plddt_file1 = afx.get_plddt_files()
      if p2:
        afx.clean()
        afx.extract_zip(p2.directory)
        plddt_file2 = afx.get_plddt_files()
        afx.plddt_results(plddt_file1[list(plddt_file1)[0]],plddt_file2[list(plddt_file2)[0]])
      else:
        afx.plddt_results(plddt_file1[list(plddt_file1)[0]])
      afx.clean()
    
    def fit(self, p2): #p2 is fit to p1
      dir_1, dir_2 = self.get_pdbs(p2)
      new_directory = afx.superimpose_proteins(dir_1,dir_2)
      afx.clean()
      os.system(f"zip -FS \"{new_directory[:-4]}.zip\" \"{new_directory}\"")
      return prediction_results(f"Superimposed {p2.name}", f"{new_directory[:-4]}.zip") #a new file with the data

    def rmsd(self, p2, start=0, end=0):
      dir_1, dir_2 = self.get_pdbs(p2)
      return afx.calc_individual_rmsd(dir_1, dir_2, start, end)

    def get_pdbs(self, p2 = 0): #not plddt nor pae
      afx.clean()
      afx.extract_zip(self.directory)
      with os.scandir("pdb_files") as pdbs:
        for pdb in pdbs:
          dir_1 = pdb.path
      if type(p2) != int:
        afx.clean()
        afx.extract_zip(p2.directory)
        with os.scandir("pdb_files") as pdbs:
          for pdb in pdbs:
            dir_2 = pdb.path
        afx.extract_zip(self.directory)
      else:
      	return dir_1
      return dir_1, dir_2

    def view(self, p2 = 0): #not finished
      if 'COLAB_GPU' in os.environ:
        print('This command is not supported by Google Colab.')
      else:
        import nglview as nv
        if type(p2) != int:
          dir_1, dir_2 = self.get_pdbs(p2)
          view = nv.show_file(dir_1)
          view.add_component(dir_2)
        else:
          dir_1 = self.get_pdbs()
          view = nv.show_file(dir_1)
        return view

    def get_molecular_weight(self):
      directory = self.get_pdbs()
      value = afx.molecular_weight(directory)
      print(value)
      return value
