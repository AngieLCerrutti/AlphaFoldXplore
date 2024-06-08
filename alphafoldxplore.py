# -*- coding: utf-8 -*-

import os
from os import name
if 'COLAB_GPU' in os.environ:
  from google.colab import files #to download the predictions later if you're on Colab
else: 
  try:
    import nglview
  except:
    pass
import subprocess
import json
#-------------------
import sys
sys.path.insert(1, 'AlphaFoldXplore')
import pickle

if "/content/tmp/bin" not in os.environ['PATH']:
  os.environ['PATH'] += ":/content/tmp/bin:/content/tmp/scripts"

from urllib import request
from concurrent import futures
import json
from matplotlib import gridspec
import zipfile
from zipfile import ZipFile
import matplotlib.pyplot as plt
import Bio
from Bio import PDB
import ipywidgets as widget
from Bio.PDB.PDBParser import PDBParser
import math
import numpy as np
import gc #free memory resources, unrelated to AlphaFold
import time
import pandas as pd
import seaborn as sns
from datetime import datetime
import shutil
import prediction_results
import re
import hashlib
os.makedirs("input", exist_ok=True)
from sys import version_info
python_version = f"{version_info.major}.{version_info.minor}"

def set_up():
  
  PYTHON_VERSION = python_version

  if not os.path.isfile("COLABFOLD_READY"):
    print("installing colabfold...")
    os.system("pip install -q --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'")
    if os.environ.get('TPU_NAME', False) != False:
      os.system("pip install -q --no-warn-conflicts -U dm-haiku==0.0.10 jax==0.3.25")
    os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabfold colabfold")
    os.system("ln -s /usr/local/lib/python3.*/dist-packages/alphafold alphafold")
    os.system("touch COLABFOLD_READY")

  if not os.path.isfile("CONDA_READY"):
    print("installing conda...")
    os.system("wget -qnc https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh")
    os.system("bash Mambaforge-Linux-x86_64.sh -bfp /usr/local")
    os.system("mamba config --set auto_update_conda false")
    os.system("touch CONDA_READY")

  if not os.path.isfile("HH_READY"):
    print("installing hhsuite...")
    os.system(f"mamba install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 python='{PYTHON_VERSION}'")
    os.system("touch HH_READY")
  if not os.path.isfile("AMBER_READY"):
    #print("installing amber...")
    #os.system(f"mamba install -y -c conda-forge openmm=7.7.0 python='{PYTHON_VERSION}' pdbfixer")
    #os.system("touch AMBER_READY")
    pass

def predict(zfile): #FASTA path inputted
  import jax #if this fails, predicion would never work either way
  protein_count = 0
  from collections import defaultdict
  d = defaultdict(str)
  with open(zfile, "r") as file1:
    for line in file1:
        if line[0] == '>': #I'm assuming header/idnumber starts with '>'
            jobname = line.strip('\n')
        else:
            sequence= line.strip('\n')
            d[jobname[1:]] += sequence
  file1.close()
  seque= []
  Z = {}
  indiv_predic_list=[]
  now = datetime.now()
  dt_string = now.strftime("%Y%m%d%H%M%S")
  zname = os.path.basename(zfile)
  zname = zname[:-6].replace(" ", "") #hopefully it had a fasta extension
  afxtname = f"{zname}_{dt_string}"
  from prediction_results import prediction_results
  for sec in d.items():
    protein_count = protein_count + 1
    start = time.time()
    import re
    #------------------------
    sequence = (str(sec[1]))
    jobname = (sec[0])
    jobname = jobname.replace("/","") # forbidden characters
    jobname = jobname.replace("\\","")
    og_jobname = jobname
    jobname = (str(jobname))
    sequence = re.sub("[^A-Z:/]", "", sequence.upper())
    sequence = re.sub(":+",":",sequence)
    sequence = re.sub("/+","/",sequence)
    sequence = re.sub("^[:/]+","",sequence)
    sequence = re.sub("[:/]+$","",sequence)

    jobname = re.sub(r'\W+', '', jobname)

    output_dir ='prediction_'+sec[0].replace(" ", "")
    output_dir = output_dir.replace("/","") # forbidden characters
    output_dir = output_dir.replace("\\","")
    os.makedirs(output_dir, exist_ok=True)
    # delete existing files in working directory
    for f in os.listdir(output_dir):
      os.remove(os.path.join(output_dir, f))
  #------------------------------
    display_images = False

    import sys
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from Bio import BiopythonDeprecationWarning
    warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)
    from pathlib import Path
    from colabfold.download import download_alphafold_params, default_data_dir
    from colabfold.utils import setup_logging
    from colabfold.batch import get_queries, run, set_model_type
    from colabfold.plot import plot_msa_v2

    import numpy as np
    try:
      K80_chk = os.popen('nvidia-smi | grep "Tesla K80" | wc -l').read()
    except:
      K80_chk = "0"
      pass
    if "1" in K80_chk:
      print("WARNING: found GPU Tesla K80: limited to total length < 1000")
      if "TF_FORCE_UNIFIED_MEMORY" in os.environ:
        del os.environ["TF_FORCE_UNIFIED_MEMORY"]
      if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
        del os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]

    from colabfold.colabfold import plot_protein
    from pathlib import Path
    import matplotlib.pyplot as plt

    # For some reason we need that to get pdbfixer to import
    if f"/usr/local/lib/python{python_version}/site-packages/" not in sys.path:
        sys.path.insert(0, f"/usr/local/lib/python{python_version}/site-packages/")

    def input_features_callback(input_features):
      if display_images:
        plot_msa_v2(input_features)
        plt.show()
        plt.close()

    def prediction_callback(protein_obj, length,
                            prediction_result, input_features, mode):
      model_name, relaxed = mode
      if not relaxed:
        if display_images:
          fig = plot_protein(protein_obj, Ls=length, dpi=150)
          plt.show()
          plt.close()

    result_dir = jobname
    log_filename = os.path.join(jobname,"log.txt")
    setup_logging(Path(log_filename))
    
    queries_path = os.path.join(jobname, f"{jobname}.csv")
    with open(queries_path, "w") as text_file:
      text_file.write(f"id,sequence\n{jobname},{sequence}")
    queries, is_complex = get_queries(queries_path)
    model_type = "alphafold2_ptm"
    model_type = set_model_type(is_complex, model_type)

    if "multimer" in model_type and max_msa is not None:
      use_cluster_profile = False
    else:
      use_cluster_profile = True

    custom_template_path = None
    use_templates = False
    num_relax = 0
    msa_mode = "mmseqs2_uniref_env"
    num_recycles= 3
    relax_max_iterations = None
    recycle_early_stop_tolerance = None
    num_seeds = 1
    use_dropout = False
    pair_mode = "unpaired"
    pairing_strategy = "greedy"
    dpi = 200 #not needed
    save_all = False
    save_recycles = False
    save_to_google_drive = False
    max_msa = None

    download_alphafold_params(model_type, Path("."))
    results = run(
        queries=queries,
        result_dir=result_dir,
        use_templates=use_templates,
        custom_template_path=custom_template_path,
        num_relax=num_relax,
        msa_mode=msa_mode,
        model_type=model_type,
        num_models=1,
        num_recycles=num_recycles,
        relax_max_iterations=relax_max_iterations,
        recycle_early_stop_tolerance=recycle_early_stop_tolerance,
        num_seeds=num_seeds,
        use_dropout=use_dropout,
        model_order=[1,2,3,4,5],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        rank_by="auto",
        pair_mode=pair_mode,
        pairing_strategy=pairing_strategy,
        stop_at_score=float(100),
        prediction_callback=prediction_callback,
        dpi=dpi,
        zip_results=False,
        save_all=save_all,
        max_msa=max_msa,
        use_cluster_profile=use_cluster_profile,
        input_features_callback=input_features_callback,
        save_recycles=save_recycles,
        user_agent="colabfold/google-colab-main",
    )
    stop = time.time()
    protein_name = str(og_jobname)
    directory = f'{afxtname}/{output_dir}.zip'
    time_spent = stop - start
    import subprocess
    machine_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used", "--format=csv,noheader"],
        encoding="utf-8", capture_output=True).stdout
    import json
    with open(os.path.join(result_dir,f'{jobname}_scores_rank_001_alphafold2_ptm_model_1_seed_000.json'),'r') as ptmstore:
      ptm = json.load(ptmstore)['ptm']
    pred_output_path = os.path.join(result_dir,f'{jobname}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb')
    pae_output_path = os.path.join(result_dir,f'{jobname}_predicted_aligned_error_v1.json')
    prediction_entry = prediction_results(protein_name,directory,time_spent,machine_info,ptm)
    Z[f'p{protein_count}'] = prediction_entry

    with open(f'{output_dir}/{og_jobname}_report.txt', 'w', encoding='utf-8') as file:
      file.write(protein_name + '\n')
      file.write(directory + '\n')
      file.write(str(time_spent) + '\n')
      file.write(machine_info)
      file.write("pTMScore=" + str(ptm) + '\n')
      file.write("version=afxl")
      file.close()
    os.system(f"cp '{pae_output_path}' '{output_dir}/{og_jobname}_pae.json'")
    os.system(f"cp '{pred_output_path}' '{output_dir}/{og_jobname}_unrelaxed.pdb'")
    os.system(f"zip -FSr {output_dir}.zip {output_dir}")
    #if 'COLAB_GPU' in os.environ:
      #files.download(f'{output_dir}.zip')
    indiv_predic_list.append(output_dir)
    shutil.rmtree(result_dir)
    print(f"Protein {protein_name} finished, proceeding...")
    continue

  with open(f'{afxtname}_list.txt', 'w', encoding='utf-8') as file:
      for result in list(Z.values()):
        file.write(f"{result.directory}\n")
      file.close()
  os.makedirs(f"{afxtname}", exist_ok=True)
  for i in range(len(list(Z.values()))):
    i1= i + 1
    olddir = Z[f'p{i1}'].directory.partition("/")[2]
    os.system(f"mv {olddir} {afxtname}")
  os.system(f"mv {afxtname}_list.txt {afxtname}")
  os.system(f"zip -FSr -D {afxtname}.zip {afxtname}")
  os.system(f"mv {afxtname}.zip {afxtname}.afxt")
  if 'COLAB_GPU' in os.environ:
    files.download(f'{afxtname}.afxt')
  print(f"Stored on your local computer. Name: \"{afxtname}.afxt'\"")
  for item in indiv_predic_list:
    shutil.rmtree(item)

  return Z

def load(filedir):
  from prediction_results import prediction_results
  Z = {}
  protein_count = 0
  extract_folder = os.path.basename(filedir[:-5])
  #os.makedirs(extract_folder, exist_ok=True)
  with ZipFile(filedir,'r') as fz:
    fz.extractall(".")
  
  if os.path.isdir(extract_folder):
    pass
  else:
    os.system(f"cp -R prediction_{extract_folder} {extract_folder}") #compatibility with old afxt files
              
  for path in os.listdir(extract_folder):
    long_path = os.path.join(extract_folder, path)
    if long_path.endswith(".txt"):
      with open(long_path,'r') as file:
        lines = file.readlines()
        file.close()
      for zipf in lines:
        zipf = zipf[:-1]
        if not "/" in zipf:
          zipf = os.path.join(extract_folder, zipf)
        if os.path.exists(zipf) == True: #Excluding linebreaks
              protein_count = protein_count + 1
              with ZipFile(zipf, 'r') as fz:
                for zip_info in fz.infolist():
                  if zip_info.filename[-1] == '/':
                    continue
                  tab = os.path.basename(zip_info.filename)
                  if tab.endswith(".txt"):
                    #zip_info.filename = os.path.basename(zip_info.filename)
                    with fz.open(zip_info.filename) as pred_info:
                      pred_lines = pred_info.readlines()
                      ptm_line = pred_lines[4].strip().decode('UTF-8')
                      pred_info.close()
                    try:
                      ptmscore = float(re.findall(r"pTMScore=?([ \d.]+)",str(ptm_line))[0])
                    except:
                      ptmscore = 0
                    prediction_entry = prediction_results(pred_lines[0].strip().decode('UTF-8'),pred_lines[1].strip().decode('UTF-8'),pred_lines[2].strip().decode('UTF-8'),pred_lines[3].strip().decode('UTF-8'),ptmscore)
                    Z[f'p{protein_count}'] = prediction_entry
  print("Loaded successfully.")
  return Z

def run():
  searching_folder = "input"
  i = 0
  while True:
    with os.scandir(searching_folder) as inputs:
      for input_sing in inputs:
        if os.path.isfile(input_sing) == True:
          input_sing = os.path.basename(input_sing)
          if input_sing.lower().endswith(".afxt"):
            print("Attempting to load a result...")
            if i==0:
              return load(f"input/{input_sing}")
            else:
              return load(f"{input_sing}")
          elif input_sing.upper().endswith(".FASTA"):
            print("Attempting to predict proteins...")
            if i==0:
              return predict(f"input/{input_sing}")
            else:
              return predict(f"{input_sing}")
    if i == 0:
      print("No file was found in the input folder. Reading from the main folder...")
      searching_folder = "."
      i += 1
    else:
      raise Exception("Error: no valid file found. (is it on the appropiate folders?)")
      
def load_af3(name, location): #esto SOLAMENTE ES PARA LOS ZIPS DE ALPHAFOLD SERVER AKA AF3
  from prediction_results import prediction_results
  Z = {}
  if os.path.exists(name):
    print(f"Error: folder with the name '{name}' already exists on this folder. Please try another name or delete/rename the folder.")
    return
 
  folder = os.path.basename(location[:-4])
  extract_folder = f'AF3_files/{folder}'
  results_folder = f'AF3_files/{folder}_fx'
  os.makedirs(results_folder, exist_ok=True)
  with ZipFile(location,'r') as fz:
    fz.extractall(extract_folder) #extract the af3 zip 
              
  for path in os.listdir(extract_folder):
    long_path = os.path.join(extract_folder, path) #let's see all its files randomly
    if long_path.endswith("_summary_confidences_0.json"): #this one has ptmscore. I use _0 for everything because nothing tells what's the best anyways
      with open(long_path,'r') as file:
        data = json.load(file)
        ptmscore = float(data['ptm'])

    if long_path.endswith("_model_0.cif"): #apparently this is a better pdb. I want the pdb anyways, we convert
      file_parser = MMCIFParser() #biopython my beloved
      structure = file_parser.get_structure("base",long_path)
      #Write PDB
      io = PDBIO()
      io.set_structure(structure)
      io.save(f'{results_folder}/{name}_relaxed.pdb') #I assume AF3 relaxes

    if long_path.endswith("_full_data_0.json"): #PAE. pLDDT was on the CIF so thiis isn't as needed anymore
      with open(long_path,'r') as file:
        data = json.load(file)
        distance = {"distance": data['pae']}
        file.close()
      with open(f'{results_folder}/{name}_pae.json', 'w', encoding='utf-8') as f: #rescuing pae alone
        json.dump(distance, f, ensure_ascii=False)
        f.close()

  #a partir de aca es toda la misma magia negra que tiene después de generar archivos con todas las otras formas.
  directory = f'{name}/{name}.zip' 
  os.makedirs(f'{name}', exist_ok=True)
  with open(f'{name}/{name}_report.txt', 'w', encoding='utf-8') as file:
    file.write(name + '\n')
    file.write(directory + '\n')
    file.write("0" + '\n')
    file.write("no info" + '\n')
    file.write("pTMScore=" + str(ptmscore) + '\n')
    file.write("version=af3")
    file.close()
  os.system(f"mv '{results_folder}/{name}_pae.json' '{name}/{name}_pae.json'")
  os.system(f"mv '{results_folder}/{name}_relaxed.pdb' '{name}/{name}_relaxed.pdb'")

  os.system(f"zip -FSr '{name}.zip' '{name}'")
  shutil.rmtree(f'{name}')
  prediction_entry = prediction_results(name,directory,"0","no info",ptmscore)
  Z[f'p1'] = prediction_entry

  os.makedirs(f'{name}', exist_ok=True) #la borre antes para purgar archivos
  with open(f'{name}/{name}_list.txt', 'w', encoding='utf-8') as file:
    for result in list(Z.values()):
      file.write(f"{result.directory}\n")
    file.close()
  os.system(f"mv '{name}.zip' '{name}/'")
  os.system(f"zip -FSr -D '{name}.zip' '{name}'")
  os.system(f"mv '{name}.zip' '{name}.afxt'")
  print(f"Stored on your local computer. Name: \"{name}.afxt'\"")
  #shutil.rmtree(f'{name}')
  return Z

###########################################################################
##The end user is not meant to use the following fuctions. For ease of simplification refer to the above ones
##and the use of the Prediction_results object.
############################################################################

def extract_zips(dir="."): #whole directory inputted
  with os.scandir(dir) as ficheros:
    for fichero in ficheros:
      if os.path.isfile(fichero) == True:
        fichero = os.path.realpath(fichero)
        if fichero.endswith(".zip"):
          with ZipFile(fichero, 'r') as fz:
            for zip_info in fz.infolist():
              if zip_info.filename[-1] == '/':
                continue
              tab = os.path.basename(zip_info.filename)
              if tab.endswith(".json"):
                zip_info.filename = os.path.basename(zip_info.filename)
                fz.extract(zip_info, "json_files")         
              elif tab.endswith(".pdb"):
                zip_info.filename = os.path.basename(zip_info.filename)
                fz.extract(zip_info, "pdb_files") 

def extract_zip(dir): #singular, zip string as parameter, must end in .zip
  if dir.endswith(".zip"):
    with ZipFile(dir, 'r') as fz:
      for zip_info in fz.infolist():
        if zip_info.filename[-1] == '/':
          continue
        tab = os.path.basename(zip_info.filename)
        if tab.endswith(".json"):
          zip_info.filename = os.path.basename(zip_info.filename)
          fz.extract(zip_info, "json_files")         
        elif tab.endswith(".pdb"):
          zip_info.filename = os.path.basename(zip_info.filename)
          if zip_info.filename.endswith("_unrelaxed.pdb"):
            extension = "_unrelaxed"
            filename = zip_info.filename[:-14]
          elif zip_info.filename.endswith("_relaxed.pdb"):
            extension = "_relaxed"
            filename = zip_info.filename[:-12]
          else: 
            extension = ""
            filename = zip_info.filename[:-4]
          if(os.path.exists(f"pdb_files/{filename}{extension}.pdb")):
            #a clone!! it must be there for a reason
            os.makedirs("pdb_files/tmp",exist_ok=True)
            fz.extract(zip_info, "pdb_files/tmp")
            iter_num = 2
            while True:
              if(os.path.exists(f"pdb_files/{filename}{extension}_{iter_num}.pdb")):
                iter_num = iter_num + 1
              else:
                break
            os.system(f"mv pdb_files/tmp/{filename}{extension}.pdb pdb_files/{filename}{extension}_{iter_num}.pdb")
            shutil.rmtree("pdb_files/tmp")
            return (f"pdb_files/{filename}{extension}_{iter_num}.pdb")
          else:
            fz.extract(zip_info, "pdb_files")
            return (f"pdb_files/{filename}{extension}.pdb")
  else:
    print("Could not extract. Zip file not found")

def clean(): #erases the folders by extract_zip and so. Meant to be used silently by the script.
  try:
    shutil.rmtree('json_files')
    shutil.rmtree('pdb_files')
  except:
    pass

def get_pae_files(dir = "json_files"): #returns a dict with pae data
  ficheros = filter(os.path.isfile, os.scandir(dir))
  ficheros = [os.path.realpath(f) for f in ficheros] # add path to each file
  ficheros.sort(key=os.path.getmtime)
  imagenes={}
  for fichero in ficheros:
    with open(fichero) as f:
      d=json.load(f)
      try:
      	dataf=(pd.DataFrame(d[0]["distance"]))
      except Exception as error:
      	dataf=(pd.DataFrame(d["distance"])) #compatibility
      imagenes[os.path.basename(fichero)] = dataf
    f.close()
  return(imagenes)

def pae_results (pae1, pae2 = 0, names=[], substract=False): 
  pae_data={}
  if type(pae2) == int: #not added
    pae2_exist = False
  else:
    pae2_exist = True
    str2 = False

  pae_data[names[0]] = pae1 #assumed to be a json dataframe

  import matplotlib as mpl
  with mpl.rc_context({'figure.figsize': [15, 6],"figure.autolayout": True}):
    df1=(pae_data[list(pae_data)[0]])
    if pae2_exist:
      pae_data[names[1]] = pae2
      df2=(pae_data[list(pae_data)[1]])
      if substract:
        fig, (ax1) = plt.subplots(ncols=1)
        df1 = df1 - df2
        cpalette = "PuOr"
      else:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        fig.subplots_adjust(wspace=0.1)
        cpalette = "plasma"
        sns.heatmap(df2, cmap=cpalette, ax=ax2)
        ax2.yaxis.tick_right()
    else:
      cpalette = "plasma"
      fig, (ax1) = plt.subplots(ncols=1)
    sns.heatmap(df1, cmap=cpalette  , ax=ax1)
    plt.show()

def get_plddt_files(dir = 'pdb_files'):
  ficheros = filter(os.path.isfile, os.scandir(dir))
  ficheros = [os.path.realpath(f) for f in ficheros] # add path to each file
  ficheros.sort(key=os.path.getmtime)
  imagenes={}
  for fichero in ficheros:
    with open(fichero) as fic:
      df67= pd.DataFrame(fic)
      dff=df67[0].str.split(expand=True)
      CAr = dff[2] == "CA" #only Alpha carbons are used
      extCA = dff[CAr][10]
      imagenes[os.path.basename(fichero)]=extCA
      fic.close()
  return(imagenes)

def plddt_results(plddt1, plddt2 = 0, names=[]):
  plddt_data={}
  label1 = names[0]
  if type(plddt2) == int: #no optional pLDDT file included
    plddt2_exist = False
    is_dict = False
  else:
    plddt2_exist = True
    if isinstance(plddt2, dict):
      is_dict = True
    else:
      is_dict = False
      label2 = names[1]

  pay1=[] #primera prot
  for m in plddt1:
    l=float(m)
    pay1.append(l)

  if plddt2_exist:
    if is_dict:
      list_of_all = []
      for plddt in plddt2.values():
        pay=[]
        for m in plddt:
          l=float(m)
          pay.append(l)
        list_of_all.append(pay)
    else:
      pay2=[] #second prot
      for m in plddt2:
        l=float(m)
        pay2.append(l)


  import matplotlib as mpl
  with mpl.rc_context({'figure.figsize': [15, 6],"figure.autolayout": True}):
    if plddt2_exist:  
      if is_dict:
        df = pd.DataFrame(list(zip(pay1)), columns = ["m1"])
        plt.plot(df['m1'], label=names[0])
        i=0
        for item in list_of_all:
          if i>0:
            df1 = pd.DataFrame(list(zip(item)), columns = ["m1"])
            plt.plot(df1['m1'], label=names[i])
          i += 1
      else:
        df1 = pd.DataFrame(list(zip(pay1,pay2)), columns = ["m1","m2"])
        plt.plot(df1['m1'], label=label1)
        plt.plot(df1['m2'], label=label2)
    else:
      df1 = pd.DataFrame(list(zip(pay1)), columns = ["m1"])
      plt.plot(df1['m1'], label=label1)
    plt.legend(loc='lower left')
    plt.xlabel('Index')
    plt.ylabel('pLDDT%')
    plt.title('pLDDT comparation between predictions')
    plt.show()
    if is_dict:
      return list_of_all

def superimpose_proteins(p1,p2, silent=False): #Protein superposition
  #Thanks to Anders Steen Christensen for the code: https://gist.github.com/andersx/6354971
  pdb_parser = Bio.PDB.PDBParser()
  ref_structure = pdb_parser.get_structure("reference", p1) 
  sample_structure = pdb_parser.get_structure("sample", p2)

  # In case personalized results are loaded and there are various models, only the first is chosen
  # change the value manually if needed!
  ref_model    = ref_structure[0]
  sample_model = sample_structure[0]

  ref_atoms = []
  sample_atoms = []

  for ref_chain in ref_model:
    for ref_res in ref_chain:
      ref_atoms.append(ref_res['CA'])

  for sample_chain in sample_model:
    for sample_res in sample_chain:
      sample_atoms.append(sample_res['CA'])

      
  #numpy works better so transforming
  ref_atoms = np.array(ref_atoms)
  sample_atoms = np.array(sample_atoms)

      
  super_imposer = Bio.PDB.Superimposer()
  super_imposer.set_atoms(ref_atoms, sample_atoms)
  super_imposer.apply(sample_structure.get_atoms()) #modifies the original variable

  if not silent:
    print('Mean RMSD from the superimposition is:')
    print (str(super_imposer.rms) + ' Å')

  io = Bio.PDB.PDBIO() # Save the superimposed protein

  io.set_structure(sample_structure) 
  og_name = os.path.basename(p2)
  io.save(f"superimposed_{og_name[:-4]}.pdb")
  return f"superimposed_{og_name[:-4]}.pdb"

def calc_individual_rmsd(p1,p2, start=0, end=0, names=[], returning="aadistance", silent=False): #for optimal results, use the superimposed protein as sample

  #Get the coordinates first

  pdb_parser = PDBParser()

  ref_structure = pdb_parser.get_structure("reference", p1) 
  # In case personalized results are loaded and there are various models, only the first is chosen
  # change the value manually if needed!
  ref_model = ref_structure[0]
  ref_atoms = []
  for ref_chain in ref_model:
    for ref_res in ref_chain:
      ref_atoms.append(ref_res['CA'])
  ref_atoms = np.array(ref_atoms)
  #p1 atoms coords
  ref_atoms_coords = np.empty((1,3))
  for atom in ref_atoms:
      ref_atoms_coords = np.append(ref_atoms_coords, np.array([atom.get_coord()]), axis=0)
  
  import matplotlib as mpl
  rmsd_list = []
  true_rmsd_list = []
  with mpl.rc_context({'figure.figsize': [15, 6],"figure.autolayout": True}):
    if isinstance(p2, list):
      amount = len(p2)
    else:
      temp = []
      temp.append("null")
      temp.append(p2)
      p2 = temp
      amount = 2 #the reference + sample
    j = 1
    while j < amount:
      try:
        sample_structure = pdb_parser.get_structure("sample", p2[j])
      except:
        sample_structure = pdb_parser.get_structure("sample", f"superimposed_{os.path.basename(p2[j])[13:]}")
      sample_model = sample_structure[0]
      sample_atoms = []
      for sample_chain in sample_model:
        for sample_res in sample_chain:
          sample_atoms.append(sample_res['CA'])
      sample_atoms = np.array(sample_atoms)
      sample_atoms_coords = np.empty((1,3))
      for atom in sample_atoms:
        sample_atoms_coords = np.append(sample_atoms_coords, np.array([atom.get_coord()]), axis=0)

      #calculate the euclidian distance/RMSD of individual carbons (and the total to check out)
      if end == 0: #if end parameter was not passed:
       end = len(ref_atoms_coords)
      i=0
      distancia_euclidiana=[] #List with distances, used for total RMSD
      rmsd_individual=[] #List with individual RMSD between pair of atoms
      for atom in ref_atoms_coords:
          if i > start and i < end:
            distancia_euclidiana.append((atom[0] - sample_atoms_coords[i][0]) * (atom[0] - sample_atoms_coords[i][0]) + (atom[1] - sample_atoms_coords[i][1]) * (atom[1] - sample_atoms_coords[i][1]) + (atom[2] - sample_atoms_coords[i][2]) * (atom[2] - sample_atoms_coords[i][2]))
          i = i + 1
          #Equivalent: sumatory of (Xip1 - Xip2)^2 where i = carbon index
      distancia_euclidiana = np.array(distancia_euclidiana)
      distancia_euclidiana = distancia_euclidiana.reshape(-1,1)
      suma = 0
      i = 0
      len_dist = len(distancia_euclidiana)
      for i in range (len(distancia_euclidiana)):
          rmsd_individual.append(math.sqrt(distancia_euclidiana[i])) #square root of every element of the list for RMSD
          suma = suma + distancia_euclidiana[i]  
      suma = math.sqrt(suma/len_dist) #Square root of the sumatory of all distances divided by protein length
      rmsd_individual = np.array(rmsd_individual)
      rmsd_individual = rmsd_individual.reshape(-1,1)
      rmsd_list.append(rmsd_individual)
      if not silent:
        plt.plot(rmsd_individual, label=f"{names[0]} + {names[j]}")
        print(f"Mean RMSD of {names[0]} + {names[j]}:")
        print(str(suma) + ' Å') #total RMSD according to formula
      j += 1
      true_rmsd_list.append(suma)
    if not silent:
      plt.legend(loc='upper left')
      plt.xlabel('Index')
      plt.ylabel('RMSD')
      plt.title('Individual RMSD between CA atoms')
      plt.show()
  if returning == "aadistance":
    return rmsd_list
  else:
    return true_rmsd_list

def calc_tmscore(p1,p2, names=[], silent=False): #for optimal results, use the superimposed protein as sample
  #Get the coordinates first
  pdb_parser = PDBParser()
  ref_structure = pdb_parser.get_structure("reference", p1) 
  # In case personalized results are loaded and there are various models, only the first is chosen
  # change the value manually if needed!
  ref_model = ref_structure[0]
  ref_atoms = []
  for ref_chain in ref_model:
    for ref_res in ref_chain:
      ref_atoms.append(ref_res['CA'])
  ref_atoms = np.array(ref_atoms)
  #p1 atoms coords
  ref_atoms_coords = np.empty((1,3))
  for atom in ref_atoms:
      ref_atoms_coords = np.append(ref_atoms_coords, np.array([atom.get_coord()]), axis=0)
  import matplotlib as mpl
  tmscore_list = []
  with mpl.rc_context({'figure.figsize': [15, 6],"figure.autolayout": True}): # if I need it?
    if isinstance(p2, list):
      amount = len(p2)
    else:
      temp = []
      temp.append("null")
      temp.append(p2)
      p2 = temp
      amount = 2 #the reference + sample
    j = 1

    while j < amount:
      try:
        sample_structure = pdb_parser.get_structure("sample", p2[j])
      except:
        sample_structure = pdb_parser.get_structure("sample", f"superimposed_{os.path.basename(p2[j])[13:]}")
      sample_model = sample_structure[0]
      sample_atoms = []
      for sample_chain in sample_model:
        for sample_res in sample_chain:
          sample_atoms.append(sample_res['CA'])
      sample_atoms = np.array(sample_atoms)
      sample_atoms_coords = np.empty((1,3))
      for atom in sample_atoms:
        sample_atoms_coords = np.append(sample_atoms_coords, np.array([atom.get_coord()]), axis=0)

      #calculate the euclidian distance of the common individual carbons (and then get the tmscore with that)
      end1 = len(ref_atoms_coords)
      end2 = len(sample_atoms_coords)
      end = min(end1,end2)
      if end1 > end2: 
        who = 1 
      elif end1 < end2: 
        who = 2
      else:
        who = 0
      i=0
      distancia_euclidiana=[] #List with distances, used for total RMSD
      for atom in ref_atoms_coords:
          if i < end:
            distancia_euclidiana.append((atom[0] - sample_atoms_coords[i][0]) * (atom[0] - sample_atoms_coords[i][0]) + (atom[1] - sample_atoms_coords[i][1]) * (atom[1] - sample_atoms_coords[i][1]) + (atom[2] - sample_atoms_coords[i][2]) * (atom[2] - sample_atoms_coords[i][2]))
          i = i + 1
          #Equivalent: sumatory of (Xip1 - Xip2)^2 where i = carbon index
      distancia_euclidiana = np.array(distancia_euclidiana)
      distancia_euclidiana = distancia_euclidiana.reshape(-1,1)
      sum_lcommon = 0
      d0_ltarget = 1.24 * np.cbrt(end2 - 15) - 1.8 #from the references
      i = 0
      len_dist = len(distancia_euclidiana) #it repeats with end, but just in case I have to make changes later
      for i in range (len_dist):
          sum_lcommon = sum_lcommon + (1 / (1 + pow(distancia_euclidiana[i]/d0_ltarget,2)))
      tm_score = float((1/end2) * sum_lcommon)
      tmscore_list.append(tm_score)
      if not silent:
        print(f"TM-score between {names[0]} + {names[j]}:")
        print(str(tm_score)) #total tm-score between 0-1 according to formula
      j += 1
  return tmscore_list

def molecular_weight(pdb):
  from Bio import SeqIO
  from Bio import SeqUtils
  record = SeqIO.read(pdb, "pdb-atom")
  return SeqUtils.molecular_weight(record.seq, "protein")
