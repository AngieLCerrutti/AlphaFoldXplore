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

    def plot_pae(self, p2=None, substract=None): #p2 must be another prediction_results object or a dict
      afx.clean()
      afx.extract_zip(self.directory)
      names = []
      names.append(self.name)
      if not substract:
        substract = False
      if p2:
        if isinstance(p2, type(self)):
          afx.extract_zip(p2.directory)
          pae_dict = afx.get_pae_files()
          names.append(p2.name)
          afx.pae_results(pae_dict[f'{self.name}_pae.json'],pae_dict[f'{p2.name}_pae.json'], names=names, substract=substract)
        else:
          pae_dict = {}
          if isinstance(p2, dict): #is it a dict?
            print("dict compatibility WIP")
            pass
            for p in p2.values():
              if p.name != self.name:
                afx.extract_zip(p.directory)
                names.append(p.name)
            pae_dict = afx.get_pae_files()
            afx.pae_results(pae_dict[f'{self.name}_pae.json'],pae_dict, names=names, substract=substract)
          else:
            print("P2 does not have the correct type. Defaulting to single plot.")
            pae_dict = afx.get_pae_files()
            afx.pae_results(pae_dict[f'{self.name}_pae.json'], names=names)
      else:
        afx.pae_results(pae_dict[f'{self.name}_pae.json'],names=names)
      afx.clean()

    def plot_plddt(self, p2=None): #p2 must be another prediction_results object
      afx.clean()
      afx.extract_zip(self.directory)
      names = []
      names.append(self.name)
      if p2:
        if isinstance(p2, type(self)):
          afx.extract_zip(p2.directory)
          plddt_dict = afx.get_plddt_files()
          names.append(p2.name)
          afx.plddt_results(plddt_dict[f'{self.name}_unrelaxed.pdb'],plddt_dict[f'{p2.name}_unrelaxed.pdb'], names=names)
        else:
          plddt_dict = {}
          if isinstance(p2, dict): #is it a dict?
            for p in p2.values():
              if p.name != self.name:
                afx.extract_zip(p.directory)
                names.append(p.name)
            plddt_dict = afx.get_plddt_files()
            afx.plddt_results(plddt_dict[f'{self.name}_unrelaxed.pdb'],plddt_dict, names=names)
          else:
            print("P2 does not have the correct type. Defaulting to single unit.")
            plddt_dict = afx.get_pae_files()
            afx.plddt_results(plddt_dict[f'{self.name}_unrelaxed.pdb'],names=names)
      else:
        plddt_dict = afx.get_pae_files()
        afx.plddt_results(plddt_dict[f'{self.name}_unrelaxed.pdb'],names=names)
      afx.clean()

    def fit(self, p2): #p2 is fit to p1
      names = []
      names.append(self.name)
      dir_1, dir_2 = self.get_pdbs(p2)
      
      if isinstance(p2, type(self)):
        new_directory = afx.superimpose_proteins(dir_1,dir_2)
        afx.clean()
        os.system(f"zip -FS \"{new_directory[:-4]}.zip\" \"{new_directory}\"")
        return prediction_results(f"Superimposed {p2.name}", f"{new_directory[:-4]}.zip") #a new file with the data
      
      elif isinstance(p2, dict):
        superimposed_dict = {}
        i = 0
        for p in p2.values():
          if p.name != self.name:
            new_directory = afx.superimpose_proteins(dir_1,dir_2[i])
            os.system(f"zip -FS \"{new_directory[:-4]}.zip\" \"{new_directory}\"")
            superimposed_dict[p.name] = prediction_results(f"Superimposed {p.name}", f"{new_directory[:-4]}.zip") #a new file with the data
            i = i + 1
          else:
            i = i + 1
        return superimposed_dict





    def rmsd(self, p2, start=0, end=0):
      names = []
      names.append(self.name)
      dir_1, dir_2 = self.get_pdbs(p2)
      if isinstance(p2, dict):
        for p in p2.values():
          if p.name != self.name:
            names.append(p.name)
      else:
        names.append(p2.name)
      return afx.calc_individual_rmsd(dir_1, dir_2, start, end, names=names)

    def get_pdbs(self, p2 = None): #not plddt nor pae
      afx.clean()
      afx.extract_zip(self.directory)
      dir_1 = f"pdb_files/{self.name}_unrelaxed.pdb"
      if p2:
        if isinstance(p2, type(self)):
          afx.extract_zip(p2.directory)
          dir_2 = f"pdb_files/{p2.name}_unrelaxed.pdb"
        else:
          if isinstance(p2, dict): #is it a dict?
            for p in p2.values():
              if p.name != self.name:
                afx.extract_zip(p.directory)
            ficheros = filter(os.path.isfile, os.scandir("pdb_files"))
            ficheros = [os.path.realpath(f) for f in ficheros] # add path to each file
            ficheros.sort(key=os.path.getmtime)
            print(ficheros)
            dir_2 = ficheros
          else:
            print("P2 does not have the correct type. Defaulting to single unit.")
            return dir_1
      else:
      	return dir_1
      return dir_1, dir_2

    def view(self, p2 = 0):
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