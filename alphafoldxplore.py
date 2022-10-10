# -*- coding: utf-8 -*-

import os
from os import name
if 'COLAB_GPU' in os.environ:
  from google.colab import files #to download the predictions later if you're on Colab
else:
  print('For best results install AlphaFoldXplore on a Colab machine.')
  try:
    import nglview
  catch:
    pass
import jax
from IPython.utils import io
import subprocess
import tqdm.notebook
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
os.makedirs("input", exist_ok=True)

def set_up():
  if 'COLAB_GPU' in os.environ:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    import jax
    if jax.local_devices()[0].platform == 'tpu':
      raise RuntimeError('Colab TPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')
    elif jax.local_devices()[0].platform == 'cpu':
      raise RuntimeError('Colab CPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')

  from IPython.utils import io
  import tqdm.notebook

  GIT_REPO = 'https://github.com/deepmind/alphafold'
  SOURCE_URL = 'https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar'
  PARAMS_DIR = './alphafold/data/params'
  PARAMS_PATH = os.path.join(PARAMS_DIR, os.path.basename(SOURCE_URL))
  TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
  
  # if not already installed
  try:
    total = 55
    with tqdm.notebook.tqdm(total=total, bar_format=TQDM_BAR_FORMAT) as pbar:
      with io.capture_output() as captured:
        if not os.path.isdir("alphafold"):
          os.system("rm -rf alphafold")
          os.system(f"git clone {GIT_REPO} alphafold")
          os.system("cd alphafold; git checkout 1d43aaff941c84dc56311076b58795797e49107b --quiet")

          # colabfold patches
          os.system("mkdir --parents tmp")
          os.system("wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/colabfold.py")
          os.system("wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/pairmsa.py")
          os.system("wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/protein.patch -P tmp/")
          os.system("wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/config.patch -P tmp/")
          os.system("wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/model.patch -P tmp/")
          os.system("wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/modules.patch -P tmp/")

          # install hhsuite
          os.system("curl -fsSL https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-SSE2-Linux.tar.gz | tar xz -C tmp/")

          # Apply multi-chain patch from Lim Heo @huhlim
          os.system("patch -u alphafold/alphafold/common/protein.py -i tmp/protein.patch")
          
          # Apply patch to dynamically control number of recycles (idea from Ryan Kibler)
          os.system("patch -u alphafold/alphafold/model/model.py -i tmp/model.patch")
          os.system("patch -u alphafold/alphafold/model/modules.py -i tmp/modules.patch")
          os.system("patch -u alphafold/alphafold/model/config.py -i tmp/config.patch")
          pbar.update(4)

          os.system("pip3 install ./alphafold")
          pbar.update(5)
        
          # speedup from kaczmarj
          os.system(f"mkdir --parents \"{PARAMS_DIR}\"")
          os.system(f"curl -fsSL \"{SOURCE_URL}\" | tar x -C \"{PARAMS_DIR}\"")
          pbar.update(14+27)

          #######################################################################
          os.system("sudo apt install --quiet --yes hmmer")
          pbar.update(3)

          # Install py3dmol.
          os.system("pip install py3dmol")
          pbar.update(1)

          # Create a ramdisk to store a database chunk to make Jackhmmer run fast.
          os.system("sudo mkdir -m 777 --parents /tmp/ramdisk")
          os.system("sudo mount -t tmpfs -o size=9G ramdisk /tmp/ramdisk")
          pbar.update(1)
        else:
          pbar.update(55)

  except subprocess.CalledProcessError:
    print(captured)
    raise

  ########################################################################################
  if "/content/tmp/bin" not in os.environ['PATH']:
    os.environ['PATH'] += ":/content/tmp/bin:/content/tmp/scripts"

def predict(zfile): #FASTA path inputted
  protein_count = 0
  TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
  if sys.version_info[1] >= 10: #if python 3.10
    import collections
    collections.Iterable = collections.abc.Iterable #solve a compatibility issue
  import colabfold as cf
  from alphafold.data import pipeline
  from alphafold.common import protein
  from alphafold.data import parsers
  import alphafold.model 
  from alphafold.model import model
  from alphafold.model import config
  from alphafold.model import data
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
  zname = zname[:-6].replace(" ", "")#hopefully it had a fasta extension
  afxtname = f"{zname}_{dt_string}"
  from prediction_results import prediction_results
  for sec in d.items():
    protein_count = protein_count + 1
    start = time.time()
    import re

    # define sequence
    #sequence = "\"sequence\"" #@param {type:"string"}
    #sequence = d.values()
    sequence= (str(sec[1]))
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

    #jobname = "ubiquitin" #@param {type:"string"}
    #jobname=d.keys()
    jobname = re.sub(r'\W+', '', jobname)

    # define number of copies
    homooligomer =  "1"
    homooligomer = re.sub("[:/]+",":",homooligomer)
    homooligomer = re.sub("^[:/]+","",homooligomer)
    homooligomer = re.sub("[:/]+$","",homooligomer)

    if len(homooligomer) == 0: homooligomer = "1"
    homooligomer = re.sub("[^0-9:]", "", homooligomer)
    homooligomers = [int(h) for h in homooligomer.split(":")]

    ori_sequence = sequence
    sequence = sequence.replace("/","").replace(":","")
    seqs = ori_sequence.replace("/","").split(":")
    if len(seqs) != len(homooligomers):
      if len(homooligomers) == 1:
        homooligomers = [homooligomers[0]] * len(seqs)
        homooligomer = ":".join([str(h) for h in homooligomers])
      else:
        while len(seqs) > len(homooligomers):
          homooligomers.append(1)
          homooligomers = homooligomers[:len(seqs)]
          homooligomer = ":".join([str(h) for h in homooligomers])
        print("WARNING: Mismatch between number of breaks ':' in 'sequence' and 'homooligomer' definition")

    full_sequence = "".join([s*h for s,h in zip(seqs,homooligomers)])

    # prediction directory
    
    output_dir ='prediction_'+sec[0].replace(" ", "")+'_'+cf.get_hash(full_sequence)[:5]
    output_dir = output_dir.replace("/","") # forbidden characters
    output_dir = output_dir.replace("\\","")
    os.makedirs(output_dir, exist_ok=True)
    # delete existing files in working directory
    for f in os.listdir(output_dir):
      os.remove(os.path.join(output_dir, f))

    MIN_SEQUENCE_LENGTH = 16
    MAX_SEQUENCE_LENGTH = 2500
    aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes
    if not set(full_sequence).issubset(aatypes):
      raise Exception(f'Input sequence contains non-amino acid letters: {set(sequence) - aatypes}. AlphaFold only supports 20 standard amino acids as inputs.')
    if len(full_sequence) < MIN_SEQUENCE_LENGTH:
      raise Exception(f'Input sequence is too short: {len(full_sequence)} amino acids, while the minimum is {MIN_SEQUENCE_LENGTH}')
    if len(full_sequence) > MAX_SEQUENCE_LENGTH:
      raise Exception(f'Input sequence is too long: {len(full_sequence)} amino acids, while the maximum is {MAX_SEQUENCE_LENGTH}. Please use the full AlphaFold system for long sequences.')

    if len(full_sequence) >= 650:
      print(f"WARNING: For a typical Google-Colab-GPU (12G) session, the max total length is ~650 residues. You are at {len(full_sequence)}! Running Alphafold may cause crashes.")

    print(f"total_length: '{len(full_sequence)}'") 
    print(f"working_directory: '{output_dir}'")    
    
    print(str(sec[1]))
    
    #for i in homooligomer:
    outp = seque.append(f"working_directory: '{output_dir}'")
    homo = seque.append(f"homooligomer: '{homooligomer}'")
    full = seque.append(f"total_length: '{(full_sequence)}'")
    full_len = seque.append(f"full_length: '{len(full_sequence)}'")
  
    msa_method = "mmseqs2" 
    add_custom_msa = False 
    msa_format = "fas" 
    pair_mode = "unpaired" 
    pair_cov = 50 
    pair_qid = 20 

    # --- Search against genetic databases ---
    os.makedirs('tmp', exist_ok=True)
    msas, deletion_matrices = [],[]

    if add_custom_msa:
      print(f"upload custom msa in '{msa_format}' format")
      msa_dict = files.upload()
      lines = msa_dict[list(msa_dict.keys())[0]].decode()
    
      # convert to a3m
      with open(f"tmp/upload.{msa_format}","w") as tmp_upload:
        tmp_upload.write(lines)
      os.system(f"reformat.pl {msa_format} a3m tmp/upload.{msa_format} tmp/upload.a3m")
      a3m_lines = open("tmp/upload.a3m","r").read()

      # parse
      msa, mtx = parsers.parse_a3m(a3m_lines)
      msas.append(msa)
      deletion_matrices.append(mtx)

      if len(msas[0][0]) != len(sequence):
        raise ValueError("ERROR: the length of msa does not match input sequence")

    if msa_method == "precomputed":
      print("upload precomputed pickled msa from previous run")
      pickled_msa_dict = files.upload()
      msas_dict = pickle.loads(pickled_msa_dict[list(pickled_msa_dict.keys())[0]])
      msas, deletion_matrices = (msas_dict[k] for k in ['msas', 'deletion_matrices'])

    elif msa_method == "single_sequence":
      if len(msas) == 0:
        msas.append([sequence])
        deletion_matrices.append([[0]*len(sequence)])

    else:
      seqs = ori_sequence.replace('/','').split(':')
      _blank_seq = ["-" * len(seq) for seq in seqs]
      _blank_mtx = [[0] * len(seq) for seq in seqs]
      def _pad(ns,vals,mode):
        if mode == "seq": _blank = _blank_seq.copy()
        if mode == "mtx": _blank = _blank_mtx.copy()
        if isinstance(ns, list):
          for n,val in zip(ns,vals): _blank[n] = val
        else: _blank[ns] = vals
        if mode == "seq": return "".join(_blank)
        if mode == "mtx": return sum(_blank,[])
      if len(seqs) == 1 or "unpaired" in pair_mode:
        # gather msas
        if msa_method == "mmseqs2":
          prefix = cf.get_hash("".join(seqs))
          prefix = os.path.join('tmp',prefix)
          print(f"running mmseqs2")
          A3M_LINES = cf.run_mmseqs2(seqs, prefix, filter=True)

        for n, seq in enumerate(seqs):
          # tmp directory
          prefix = cf.get_hash(seq)
          prefix = os.path.join('tmp',prefix)

          if msa_method == "mmseqs2":
            # run mmseqs2
            a3m_lines = A3M_LINES[n]
            msa, mtx = parsers.parse_a3m(a3m_lines)
            msas_, mtxs_ = [msa],[mtx]
        
          # pad sequences
          for msa_,mtx_ in zip(msas_,mtxs_):
            msa,mtx = [sequence],[[0]*len(sequence)]      
            for s,m in zip(msa_,mtx_):
              msa.append(_pad(n,s,"seq"))
              mtx.append(_pad(n,m,"mtx"))

            msas.append(msa)
            deletion_matrices.append(mtx)

  
    num_relax = "None"
    rank_by = "pLDDT"
    use_turbo = True
    max_msa = "64:128"
    max_msa_clusters, max_extra_msa = [int(x) for x in max_msa.split(":")]

    show_images = False

    num_models = 1 #cambiar si se quiere usar más modelos de template para predicciones
    use_ptm = True 
    num_ensemble = 1 
    max_recycles = 1 
    tol = 0.1 
    is_training = True 
    num_samples = 1 
    subsample_msa = True 
    
    save_pae_json = False
    save_tmp_pdb = False


    if use_ptm == False and rank_by == "pTMscore":
      print("WARNING: models will be ranked by pLDDT, 'use_ptm' is needed to compute pTMscore")
      rank_by = "pLDDT"

    #############################
    # delete old files
    #############################
    for f in os.listdir(output_dir):
      if "rank_" in f:
        os.remove(os.path.join(output_dir, f))

    #############################
    # homooligomerize
    #############################
    lengths = [len(seq) for seq in seqs]
    msas_mod, deletion_matrices_mod = cf.homooligomerize_heterooligomer(msas, deletion_matrices,
                                                                        lengths, homooligomers)
    #############################
    # define input features
    #############################
    def _placeholder_template_feats(num_templates_, num_res_):
      return {
          'template_aatype': np.zeros([num_templates_, num_res_, 22], np.float32),
          'template_all_atom_masks': np.zeros([num_templates_, num_res_, 37, 3], np.float32),
          'template_all_atom_positions': np.zeros([num_templates_, num_res_, 37], np.float32),
          'template_domain_names': np.zeros([num_templates_], np.float32),
          'template_sum_probs': np.zeros([num_templates_], np.float32),
      }

    num_res = len(full_sequence)
    feature_dict = {}
    feature_dict.update(pipeline.make_sequence_features(full_sequence, 'test', num_res))
    feature_dict.update(pipeline.make_msa_features(msas_mod, deletion_matrices=deletion_matrices_mod))
    if not use_turbo:
      feature_dict.update(_placeholder_template_feats(0, num_res))

    def do_subsample_msa(F, random_seed=0):
      '''subsample msa to avoid running out of memory'''
      N = len(F["msa"])
      L = len(F["residue_index"])
      N_ = int(3E7/L)
      if N > N_:
        print(f"whhhaaa... too many sequences ({N}) subsampling to {N_}")
        np.random.seed(random_seed)
        idx = np.append(0,np.random.permutation(np.arange(1,N)))[:N_]
        F_ = {}
        F_["msa"] = F["msa"][idx]
        F_["deletion_matrix_int"] = F["deletion_matrix_int"][idx]
        F_["num_alignments"] = np.full_like(F["num_alignments"],N_)
        for k in ['aatype', 'between_segment_residues',
                  'domain_name', 'residue_index',
                  'seq_length', 'sequence']:
                  F_[k] = F[k]
        return F_
      else:
        return F

    ################################
    # set chain breaks
    ################################
    Ls = []
    for seq,h in zip(ori_sequence.split(":"),homooligomers):
      Ls += [len(s) for s in seq.split("/")] * h
    Ls_plot = sum([[len(seq)]*h for seq,h in zip(seqs,homooligomers)],[])
    feature_dict['residue_index'] = cf.chain_break(feature_dict['residue_index'], Ls)

    ###########################
    # run alphafold
    ###########################
    def parse_results(prediction_result, processed_feature_dict):
      b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']  
      dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
      dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
      contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[:,:,dist_bins < 8].sum(-1)

      out = {"unrelaxed_protein": protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors),
            "plddt": prediction_result['plddt'],
            "pLDDT": prediction_result['plddt'].mean(),
            "dists": dist_mtx,
            "adj": contact_mtx}

      if "ptm" in prediction_result:
        out.update({"pae": prediction_result['predicted_aligned_error'],
                    "pTMscore": prediction_result['ptm']})
  
      return out    
  
    model_names = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5'][:num_models]
    total = len(model_names) * num_samples
    with tqdm.notebook.tqdm(total=total, bar_format=TQDM_BAR_FORMAT) as pbar:

  
      #######################################################################
      # precompile model and recompile only if length changes
      #######################################################################
      if use_turbo:
        name = "model_5_ptm" if use_ptm else "model_5"
        N = len(feature_dict["msa"])
        L = len(feature_dict["residue_index"])
        compiled = (N, L, use_ptm, max_recycles, tol, num_ensemble, max_msa, is_training)
        if "COMPILED" in dir():
          if COMPILED != compiled: recompile = True
        else: recompile = True
        if recompile:
          cf.clear_mem("gpu")
          cfg = config.model_config(name)      

          # set size of msa (to reduce memory requirements)
          msa_clusters = min(N, max_msa_clusters)
          cfg.data.eval.max_msa_clusters = msa_clusters
          cfg.data.common.max_extra_msa = max(min(N-msa_clusters,max_extra_msa),1)

          cfg.data.common.num_recycle = max_recycles
          cfg.model.num_recycle = max_recycles
          cfg.model.recycle_tol = tol
          cfg.data.eval.num_ensemble = num_ensemble

          params = data.get_model_haiku_params(name,'./alphafold/data')
          model_runner = model.RunModel(cfg, params, is_training=is_training)
          COMPILED = compiled
          recompile = False

      else:
        cf.clear_mem("gpu")
        recompile = True

      # cleanup
      if "outs" in dir(): del outs
      outs = {}
      cf.clear_mem("cpu")  

      #######################################################################
      def report(key):
        pbar.update(n=1)
        o = outs[key]
        line = f"{key} recycles:{o['recycles']} tol:{o['tol']:.2f} pLDDT:{o['pLDDT']:.2f}"
        if use_ptm: line += f" pTMscore:{o['pTMscore']:.2f}"
        print(line)
        if show_images:
          fig = cf.plot_protein(o['unrelaxed_protein'], Ls=Ls_plot, dpi=100)
          plt.show()
        if save_tmp_pdb:
          tmp_pdb_path = os.path.join(output_dir,f'unranked_{key}_{jobname}_unrelaxed.pdb')
          pdb_lines = protein.to_pdb(o['unrelaxed_protein'])
          with open(tmp_pdb_path, 'w') as f: f.write(pdb_lines)

      if use_turbo:
        # go through each random_seed
        import warnings
        with warnings.catch_warnings():
          warnings.simplefilter(action='ignore', category=FutureWarning) #Jax is annoying so I temporarily do this. Version is specified either way
          for seed in range(num_samples):
          
            # prep input features
            if subsample_msa:
              sampled_feats_dict = do_subsample_msa(feature_dict, random_seed=seed)    
              processed_feature_dict = model_runner.process_features(sampled_feats_dict, random_seed=seed)
            else:
              processed_feature_dict = model_runner.process_features(feature_dict, random_seed=seed)

            # go through each model
            for num, model_name in enumerate(model_names):
              name = model_name+"_ptm" if use_ptm else model_name
              key = f"{name}_seed_{seed}"
              pbar.set_description(f'Running {key}')

              # replace model parameters
              params = data.get_model_haiku_params(name, './alphafold/data')
              for k in model_runner.params.keys():
                model_runner.params[k] = params[k]

              # predict
              prediction_result, (r, t) = cf.to(model_runner.predict(processed_feature_dict, random_seed=seed),"cpu")

              # save results
              outs[key] = parse_results(prediction_result, processed_feature_dict)
              outs[key].update({"recycles":r, "tol":t})
              report(key)

              del prediction_result, params
            del sampled_feats_dict, processed_feature_dict

      else:  
        # go through each model
        for num, model_name in enumerate(model_names):
          name = model_name+"_ptm" if use_ptm else model_name
          params = data.get_model_haiku_params(name, './alphafold/data')  
          cfg = config.model_config(name)
          cfg.data.common.num_recycle = cfg.model.num_recycle = max_recycles
          cfg.model.recycle_tol = tol
          cfg.data.eval.num_ensemble = num_ensemble
          model_runner = model.RunModel(cfg, params, is_training=is_training)

          # go through each random_seed
          for seed in range(num_samples):
            key = f"{name}_seed_{seed}"
            pbar.set_description(f'Running {key}')
            processed_feature_dict = model_runner.process_features(feature_dict, random_seed=seed)
            prediction_result, (r, t) = cf.to(model_runner.predict(processed_feature_dict, random_seed=seed),"cpu")
            outs[key] = parse_results(prediction_result, processed_feature_dict)
            outs[key].update({"recycles":r, "tol":t})
            report(key)

            # cleanup
            del processed_feature_dict, prediction_result

          del params, model_runner, cfg
          cf.clear_mem("gpu")

      # delete old files
      for f in os.listdir(output_dir):
        if "rank" in f:
          os.remove(os.path.join(output_dir, f))

      # Find the best model according to the mean pLDDT.
      model_rank = list(outs.keys())
      model_rank = [model_rank[i] for i in np.argsort([outs[x][rank_by] for x in model_rank])[::-1]]

      # Write out the prediction
      for n,key in enumerate(model_rank):
        prefix = f"rank_{n+1}_{key}" 
        pred_output_path = os.path.join(output_dir,f'{og_jobname}_unrelaxed.pdb')
        #fig = cf.plot_protein(outs[key]["unrelaxed_protein"], Ls=Ls_plot, dpi=200)
        #plt.savefig(os.path.join(output_dir,f'{prefix}.png'), bbox_inches = 'tight')
        #plt.close(fig)

        pdb_lines = protein.to_pdb(outs[key]["unrelaxed_protein"])
        with open(pred_output_path, 'w') as f:
          f.write(pdb_lines)
        
    ############################################################
    print(f"model rank based on {rank_by}")

    for n,key in enumerate(model_rank):    
        pae = outs[key]["pae"]
        max_pae = pae.max()
        pae_output_path = os.path.join(output_dir,f'{og_jobname}_pae.json')
        rounded_errors = np.round(np.array(pae), decimals= 1)
        indices = np.indices((len(rounded_errors), len(rounded_errors))) + 1
        indices_1 = indices[0].flatten().tolist()
        indices_2 = indices[1].flatten().tolist()
        pae_data = json.dumps([{
            'residue1': indices_1,
            'residue2': indices_2,
            'distance': rounded_errors.tolist(),
            'max_predicted_aligned_error': max_pae.item()
        }],
                                indent=None,
                                separators=(',', ':'))

        with open(pae_output_path, 'w') as f:
          f.write(pae_data)

    #shutil.make_archive(output_dir, "zip", os.getcwd())
      #for n,key in enumerate(model_rank):
      #print(f"rank_{n+1}_{key} {rank_by}:{outs[key][rank_by]:.2f}")

    stop = time.time()
    protein_name = str(og_jobname)
    directory = f'{afxtname}/{output_dir}.zip'
    time_spent = stop - start
    machine_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used", "--format=csv,noheader"],
        encoding="utf-8", capture_output=True).stdout
    prediction_entry = prediction_results(protein_name,directory,time_spent,machine_info)
    Z[f'p{protein_count}'] = prediction_entry

    with open(f'{output_dir}/{og_jobname}_report.txt', 'w', encoding='utf-8') as file:
      file.write(protein_name + '\n')
      file.write(directory + '\n')
      file.write(str(time_spent) + '\n')
      file.write(machine_info)
      file.close()
    os.system(f"zip -FSr {output_dir}.zip {output_dir}")
    #if 'COLAB_GPU' in os.environ:
      #files.download(f'{output_dir}.zip')
    indiv_predic_list.append(output_dir)
    print(f"Protein {protein_name} finished, proceeding...")
    gc.collect() #free memory to be able to run further iterations
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
                      pred_info.close()
                    #details = pred_lines.values()
                    prediction_entry = prediction_results(pred_lines[0].strip().decode('UTF-8'),pred_lines[1].strip().decode('UTF-8'),pred_lines[2].strip().decode('UTF-8'),pred_lines[3].strip().decode('UTF-8'))
                    Z[f'p{protein_count}'] = prediction_entry
  print("Loaded successfully.")
  return Z

def run():
  with os.scandir("input") as inputs:
    for input_sing in inputs:
      if os.path.isfile(input_sing) == True:
        input_sing = os.path.basename(input_sing)
        if input_sing.lower().endswith(".afxt"):
          print("Attempting to load a result...")
          return load(f"input/{input_sing}")
        elif input_sing.upper().endswith(".FASTA"):
          print("Attempting to predict proteins...")
          return predict(f"input/{input_sing}")

  print("No file was found in the input folder. Reading from the main folder...")

  with os.scandir(".") as inputs:
    for input_sing in inputs:
      if os.path.isfile(input_sing) == True:
        input_sing = os.path.basename(input_sing)
        if input_sing.lower().endswith(".afxt"):
          return load(f"{input_sing}")
        elif input_sing.endswith(".FASTA"):
          return predict(f"{input_sing}")
  raise Exception("Error: no valid file found.")

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
          fz.extract(zip_info, "pdb_files")
  else:
    print("Could not extract. Zip file not found")

def clean(): #erases the folders by extract_zip and so. Meant to be used silently by the script.
  try:
    shutil.rmtree('json_files')
    shutil.rmtree('pdb_files')
  except:
    pass

def get_pae_files(dir = "json_files"): #returns a dict with pae data
  with os.scandir(dir) as ficheros:
    imagenes={}
    for fichero in ficheros:
      if os.path.isfile(fichero) == True:
        with open(fichero) as f:
          d=json.load(f)
          dataf=(pd.DataFrame(d[0]["distance"]))
          imagenes[os.path.basename(fichero)] = dataf
        f.close()
  return(imagenes)

def pae_results (pae1, pae2 = 0, substract=False): # two strings with dir to pae files, pae2 is optional. Also admits dataframes
  pae_data={}
  if type(pae1) == str:
    str1 = True
  else:
    str1 = False

  if type(pae2) == int:
    pae2_exist = False
  else:
    pae2_exist = True
    if type(pae2) == str:
      str2 = True
    else:
      str2 = False

  if str1:
    f1 = open(pae1)
    d1=json.load(f1)
    f1.close()
    dataf1=(pd.DataFrame(d1[0]["distance"]))
    pae_data[os.path.basename(pae1)] = dataf1
  else:
    pae_data["protein 1"] = pae1 #assumed to be a json dataframe

  import matplotlib as mpl
  with mpl.rc_context({'figure.figsize': [15, 6],"figure.autolayout": True}):
    df1=(pae_data[list(pae_data)[0]])
    if pae2_exist:
      if str2:
        f2 = open(pae2)
        d2=json.load(f2)
        f2.close()
        dataf2=(pd.DataFrame(d2[0]["distance"]))
        pae_data[os.path.basename(pae2)] = dataf2
      else:
        pae_data["Protein 2"] = pae2
      df2=(pae_data[list(pae_data)[1]])
      if substract:
        fig, (ax1) = plt.subplots(ncols=1)
        df1 = df1 - df2
      else:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        fig.subplots_adjust(wspace=0.1)
        sns.heatmap(df2, cmap="plasma", ax=ax2)
        ax2.yaxis.tick_right()
    else:
      fig, (ax1) = plt.subplots(ncols=1)
    sns.heatmap(df1, cmap="plasma"  , ax=ax1)
    plt.show()

def get_plddt_files(dir = 'pdb_files'):
  with os.scandir(dir) as ficheros:
    imagenes={}
    for fichero in ficheros:
      if os.path.isfile(fichero) == True:
        with open(fichero) as fic:
          df67= pd.DataFrame(fic)
          dff=df67[0].str.split(expand=True)
          CAr = dff[2] == "CA" #only Alpha carbons are used
          extCA = dff[CAr][10]
          imagenes[os.path.basename(fichero)]=extCA
          fic.close()
  return(imagenes)

def plddt_results(plddt1, plddt2 = 0):
  plddt_data={}
  if type(plddt1) == str:
    str1 = True
    label1 = os.path.basename(plddt1)
  else:
    str1 = False
    label1 = 'protein 1'

  if type(plddt2) == int: #no optional pLDDT file included
    plddt2_exist = False
  else:
    plddt2_exist = True
    if type(plddt2) == str:
      str2 = True
      label2 = os.path.basename(plddt2)
    else:
      str2 = False
      label2 = 'protein 2'

  if str1:
    with open(plddt1) as f1:
      df1=pd.DataFrame(f1)
      dff1=df1[0].str.split(expand=True)
      CAr1 = dff1[2] == "CA" #only Alpha carbons are used
      extCA1 = dff1[CAr1][10]
      plddt_data[label1]=extCA1
      f1.close()
  else:
    plddt_data[label1]=plddt1
  pay1=[] #primera prot
  for m in plddt_data[list(plddt_data)[0]]:
    l=float(m)
    pay1.append(l)

  if plddt2_exist:
    if str2:
      with open(plddt2) as f2:
        df2= pd.DataFrame(f2)
        dff2=df2[0].str.split(expand=True)
        CAr2 = dff2[2] == "CA" #only Alpha carbons are used
        extCA2 = dff2[CAr2][10]
        plddt_data[label2]=extCA2
        f2.close()
    else:
      plddt_data[label2]=plddt2
    pay2=[] #second prot
    for m in plddt_data[list(plddt_data)[1]]:
      l=float(m)
      pay2.append(l)
  import matplotlib as mpl
  with mpl.rc_context({'figure.figsize': [15, 6],"figure.autolayout": True}):
    if plddt2_exist:  
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

def superimpose_proteins(p1,p2): #Protein superposition
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

  print('Mean RMSD from the superimposition is:')
  print (str(super_imposer.rms) + ' Å')

  io = Bio.PDB.PDBIO() # Save the superimposed protein

  io.set_structure(sample_structure) 
  og_name = os.path.basename(p2)
  io.save(f"superimposed_{og_name[:-4]}.pdb")
  return f"superimposed_{og_name[:-4]}.pdb"

def calc_individual_rmsd(p1,p2, start=0, end=0): #for optimal results, use the superimposed protein as sample

  #Get the coordinates first

  pdb_parser = PDBParser()

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

  ref_atoms = np.array(ref_atoms)
  sample_atoms = np.array(sample_atoms)

  #p1 atoms coords
  ref_atoms_coords = np.empty((1,3))
  for atom in ref_atoms:
      ref_atoms_coords = np.append(ref_atoms_coords, np.array([atom.get_coord()]), axis=0)

  #p2 atoms coords
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
  import matplotlib as mpl
  with mpl.rc_context({'figure.figsize': [15, 6],"figure.autolayout": True}):
    plt.plot(rmsd_individual, label=f"{os.path.basename(p1)} + {os.path.basename(p2)}")
    plt.legend(loc='upper left')
    plt.xlabel('Index')
    plt.ylabel('RMSD')
    plt.title('Individual RMSD between CA atoms')
    plt.show()
    print("Mean RMSD:")
    print(str(suma) + ' Å') #total RMSD according to formula
  return rmsd_individual

def molecular_weight(pdb):
  from Bio import SeqIO
  from Bio import SeqUtils
  record = SeqIO.read(pdb, "pdb-atom")
  return SeqUtils.molecular_weight(record.seq, "protein")
