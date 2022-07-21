# -*- coding: utf-8 -*-

import os
from os import name
if 'COLAB_GPU' in os.environ:
  from google.colab import files #to download the predictions later if you're on Colab
else:
  raise RuntimeError('Non-Colab devices not supported. Please install AlphaFoldXplore on a Colab machine.')
import jax
from IPython.utils import io
import subprocess
import tqdm.notebook
import json
#-------------------
import sys
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


  
# Commented out IPython magic to ensure Python compatibility.
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
  # --- Python imports ---
  import colabfold as cf
  import pairmsa
  import sys
  import pickle

  if "/content/tmp/bin" not in os.environ['PATH']:
    os.environ['PATH'] += ":/content/tmp/bin:/content/tmp/scripts"

  from urllib import request
  from concurrent import futures
  import json
  from matplotlib import gridspec
  import matplotlib.pyplot as plt
  from Bio import PDB
  import ipywidgets as widget
  from Bio.PDB.PDBParser import PDBParser
  import math
  import numpy as np
  import py3Dmol
  import gc #free memory resources, unrelated to AlphaFold
  import time #to pause, unrelated to AlphaFold

  from alphafold.model import model
  from alphafold.model import config
  from alphafold.model import data

  from alphafold.data import parsers
  from alphafold.data import pipeline

  from alphafold.common import protein

class prediction_result:
  def __init__(self):
    print("alive")
    self.dirdict = {}

  def add_entry(self, name, data):
    self.dirdict[name] = data


def predict(zfile): #se le pasa la dirección a un archivo FASTA
  TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
  import colabfold as cf
  from alphafold.data import pipeline
  from alphafold.common import protein
  from alphafold.data import parsers
  import alphafold.model #reimportando lo que no se podía aimportar al principio por tener que descargar AlphaFold
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
  class prediction_result:
    def __init__(self):
      print("alive")
      self.dirdict = {}

    def add_entry(self, name, data):
      self.dirdict[name] = data
  Z = prediction_result()
  for sec in d.items():
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

    if len(full_sequence) > 700:
      print(f"WARNING: For a typical Google-Colab-GPU (12G) session, the max total length is ~700 residues. You are at {len(full_sequence)}! Running Alphafold may cause crashes.")

    print(f"homooligomer: '{homooligomer}'")
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

    ####################################################################################
    # PAIR_MSA
    ####################################################################################
      if len(seqs) > 1 and (pair_mode == "paired" or pair_mode == "unpaired+paired"):
        print("attempting to pair some sequences...")

        if msa_method == "mmseqs2":
          prefix = cf.get_hash("".join(seqs))
          prefix = os.path.join('tmp',prefix)
          print(f"running mmseqs2_noenv_nofilter on all seqs")
          A3M_LINES = cf.run_mmseqs2(seqs, prefix, use_env=False, use_filter=False)

        _data = []
        for a in range(len(seqs)):
          print(f"prepping seq_{a}")
          _seq = seqs[a]
          _prefix = os.path.join('tmp',cf.get_hash(_seq))

          if msa_method == "mmseqs2":
            a3m_lines = A3M_LINES[a]
            _msa, _mtx, _lab = pairmsa.parse_a3m(a3m_lines,
                                              filter_qid=pair_qid/100,
                                              filter_cov=pair_cov/100)
          if len(_msa) > 1:
            _data.append(pairmsa.hash_it(_msa, _lab, _mtx, call_uniprot=False))
          else:
            _data.append(None)
      
        Ln = len(seqs)
        O = [[None for _ in seqs] for _ in seqs]
        for a in range(Ln):
          if _data[a] is not None:
            for b in range(a+1,Ln):
              if _data[b] is not None:
                print(f"attempting pairwise stitch for {a} {b}")            
                O[a][b] = pairmsa._stitch(_data[a],_data[b])
                _seq_a, _seq_b, _mtx_a, _mtx_b = (*O[a][b]["seq"],*O[a][b]["mtx"])

                ##############################################
                # filter to remove redundant sequences
                ##############################################
                ok = []
                with open("tmp/tmp.fas","w") as fas_file:
                  fas_file.writelines([f">{n}\n{a+b}\n" for n,(a,b) in enumerate(zip(_seq_a,_seq_b))])
                os.system("hhfilter -maxseq 1000000 -i tmp/tmp.fas -o tmp/tmp.id90.fas -id 90")
                for line in open("tmp/tmp.id90.fas","r"):
                  if line.startswith(">"): ok.append(int(line[1:]))
                ##############################################            
                print(f"found {len(_seq_a)} pairs ({len(ok)} after filtering)")

                if len(_seq_a) > 0:
                  msa,mtx = [sequence],[[0]*len(sequence)]
                  for s_a,s_b,m_a,m_b in zip(_seq_a, _seq_b, _mtx_a, _mtx_b):
                    msa.append(_pad([a,b],[s_a,s_b],"seq"))
                    mtx.append(_pad([a,b],[m_a,m_b],"mtx"))
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
    prediction_entry = {}
    prediction_entry['pae'] = pae_output_path
    prediction_entry['plddt'] = pred_output_path
    prediction_entry['time'] = stop - start
    machine_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used", "--format=csv,noheader"],
        encoding="utf-8", capture_output=True).stdout
    prediction_entry['machine_details'] = machine_info
    
    Z.add_entry(output_dir, prediction_entry)
    os.system(f"zip -FSr {output_dir}.zip {output_dir}")
    files.download(f'{output_dir}.zip')


    gc.collect() #free memory to be able to run further iterations
    print(f"Protein finished, proceeding...")
    continue
  extract_zips()
  return Z


def extract_zips(dir="."):
  with os.scandir(dir) as ficheros:
    for fichero in ficheros:
      if os.path.isfile(fichero) == True:
        fichero = os.path.basename(fichero)
        if fichero.endswith(".zip"):
          with ZipFile(fichero, 'r') as fz:
            for zip_info in fz.infolist():
              if zip_info.filename[-1] == '/':
                continue
              tab = os.path.basename(zip_info.filename)
              if tab.endswith(".json"):
                zip_info.filename = os.path.basename(zip_info.filename)
                lista1=fz.extract(zip_info, "json_files")         
                
          with ZipFile(fichero, 'r') as fz:
            for zip_info in fz.infolist():
              if zip_info.filename[-1] == '/':
                continue
              tab = os.path.basename(zip_info.filename)
              if tab.endswith(".pdb"):
                zip_info.filename = os.path.basename(zip_info.filename)
                lista2=fz.extract(zip_info, "pdb_files") 

def get_pae_files(dir = "json_files"): #returns a dict with pae data
  with os.scandir(dir) as ficheros:
    imagenes={}
    for fichero in ficheros:
      if os.path.isfile(fichero) == True:
        with open(fichero) as f:
          d=json.load(f)
          dataf=(pd.DataFrame(d[0]["distance"]))
          imagenes[os.path.basename(fichero)] = dataf
        #fin for
        f.close()
  return(imagenes)

def pae_results (pae1, pae2 = 0): # dos strings con direcciones a archivos pae, pae2 es opcional. también admite dataframes
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
    pae_data["protein 1"] = pae1 #se asume que es un dataframe json

  plt.rcParams["figure.figsize"] = [15, 4]
  plt.rcParams["figure.autolayout"] = True
  df1=(pae_data[list(pae_data)[0]])
  fig, (ax1, ax2) = plt.subplots(ncols=2)
  fig.subplots_adjust(wspace=0.1)
  sns.heatmap(df1, cmap="plasma"  , ax=ax1) #cbar=False
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
    sns.heatmap(df2, cmap="viridis", ax=ax2)
    ax2.yaxis.tick_right()
  plt.show()

def get_plddt_files(dir = 'pdb_files'):
  with os.scandir(dir) as ficheros:
    imagenes={}
    for fichero in ficheros:
      if os.path.isfile(fichero) == True:
        with open(fichero) as fic:
          df67= pd.DataFrame(fic)
          dff=df67[0].str.split(expand=True)
          CAr = dff[2] == "CA" #solo carbonos Alfa usados
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

  if type(plddt2) == int: #el 0 que se le dio 
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
      CAr1 = dff1[2] == "CA" #solo carbonos Alfa usados
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
        CAr2 = dff2[2] == "CA" #solo carbonos Alfa usados
        extCA2 = dff2[CAr2][10]
        plddt_data[label2]=extCA2
        f2.close()
    else:
      plddt_data[label2]=plddt2
    pay2=[] #segunda prot
    for m in plddt_data[list(plddt_data)[1]]:
      l=float(m)
      pay2.append(l)

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

def superimpose_proteins(p1,p2): #Superposición de proteínas
  #Agradecimientos a Anders Steen Christensen por el código: https://gist.github.com/andersx/6354971
  pdb_parser = Bio.PDB.PDBParser() # Iniciar el parser
  # Conseguir las estructuras
  ref_structure = pdb_parser.get_structure("reference", p1) 
  sample_structure = pdb_parser.get_structure("sample", p2)

  # Por si acaso hay varios modelos, se elige el primero de cada pdb
  # cambiar el valor de ser necesario manualmente!
  ref_model    = ref_structure[0]
  sample_model = sample_structure[0]

  # Se hace una lista de los átomos (en las estructuras) que se desean alinear.
  # En este caso se usan los átomos CA, Carbono Alfa
  ref_atoms = []
  sample_atoms = []

  for ref_chain in ref_model:
    for ref_res in ref_chain:
      # Agregar los CA a la lista
      ref_atoms.append(ref_res['CA'])

  # Hacer lo mismo para la estructura a modificar
  for sample_chain in sample_model:
    for sample_res in sample_chain:
      sample_atoms.append(sample_res['CA'])

      
  #Pasando los arreglos a las versiones de numpy
  ref_atoms = np.array(ref_atoms)
  sample_atoms = np.array(sample_atoms)

      
  # Iniciamiento del superposicionador:
  super_imposer = Bio.PDB.Superimposer()
  super_imposer.set_atoms(ref_atoms, sample_atoms)
  super_imposer.apply(sample_structure.get_atoms()) #afecta a la estructura original

  # Impresión del RMSD:
  print('Mean RMSD from the superimposition is:')
  print (str(super_imposer.rms) + ' Å')

  # Almacenar la versión alineada de la proteína de ejemplo
  io = Bio.PDB.PDBIO()

  i = 0
  io.set_structure(sample_structure) 
  io.save(f"superimposed_{os.path.basename(p2)}.pdb")

def calc_individual_rmsd(p1,p2, start=0, end=0): #para resultados óptimos, utilizar la proteína superpuesta como parámetro en p2
 #devuelve la lista con rmsd

  #Para calcular el RMSD de carbonos individuales, es necesario extraer las coordenadas de ellos primero

  pdb_parser = PDBParser() # Iniciar el parser

  # Conseguir las estructuras
  ref_structure = pdb_parser.get_structure("reference", p1) 
  sample_structure = pdb_parser.get_structure("sample", p2)

  # Por si acaso hay varios modelos, se elige el primero de cada pdb
  # cambiar el valor de ser necesario manualmente! (o pasar como parámetro, quién sabe)
  ref_model    = ref_structure[0]
  sample_model = sample_structure[0]

  # Se hace una lista de los átomos (en las estructuras) que se desean alinear.
  # En este caso se usan los átomos CA, Carbono Alfa
  ref_atoms = []
  sample_atoms = []

  for ref_chain in ref_model:
    for ref_res in ref_chain:
      # Agregar los CA a la lista
      ref_atoms.append(ref_res['CA'])

  # Hacer lo mismo para la estructura de p2
  for sample_chain in sample_model:
    for sample_res in sample_chain:
      sample_atoms.append(sample_res['CA'])

      
  #Pasando los arreglos a las versiones de numpy
  ref_atoms = np.array(ref_atoms)
  sample_atoms = np.array(sample_atoms)


  #Los carbonos de la proteína p1
  ref_atoms_coords = np.empty((1,3))
  for atom in ref_atoms:
      ref_atoms_coords = np.append(ref_atoms_coords, np.array([atom.get_coord()]), axis=0)

  #Los carbonos de la proteína p2
  sample_atoms_coords = np.empty((1,3))
  for atom in sample_atoms:
      sample_atoms_coords = np.append(sample_atoms_coords, np.array([atom.get_coord()]), axis=0)
      
  #calcular la distancia euclidiana/RMSD de carbonos individuales (y también el total para corroborar)
  if end == 0: #si no se pasó el parámetro de end:
   end = len(ref_atoms_coords)
  i=0
  distancia_euclidiana=[] #Lista con las distancias, que será usada luego para el RMSD total
  rmsd_individual=[] #Lista que contendrá las RMSD entre pares de carbonos individualmente
  for atom in ref_atoms_coords:
      if i > start and i < end:
        distancia_euclidiana.append((atom[0] - sample_atoms_coords[i][0]) * (atom[0] - sample_atoms_coords[i][0]) + (atom[1] - sample_atoms_coords[i][1]) * (atom[1] - sample_atoms_coords[i][1]) + (atom[2] - sample_atoms_coords[i][2]) * (atom[2] - sample_atoms_coords[i][2]))
      i = i + 1
      #En equivalencia: sumatoria de (Xip1 - Xip2)^2 donde i se refiere al index de los carbonos
  distancia_euclidiana = np.array(distancia_euclidiana)
  distancia_euclidiana = distancia_euclidiana.reshape(-1,1)
  suma = 0
  i = 0
  len_dist = len(distancia_euclidiana)
  for i in range (len(distancia_euclidiana)):
      rmsd_individual.append(math.sqrt(distancia_euclidiana[i])) #Raiz cuadrada de cada elemento de la lista para el RMSD
      suma = suma + distancia_euclidiana[i] 
      
  suma = math.sqrt(suma/len_dist) #Raiz cuadrada de la sumatoria de todas las distancias dividida la longitud de las proteínas
  rmsd_individual = np.array(rmsd_individual)
  rmsd_individual = rmsd_individual.reshape(-1,1)
  plt.plot(rmsd_individual, label=f"{os.path.basename(p1)} + {os.path.basename(p2)}")
  plt.legend(loc='upper left')
  plt.xlabel('Index')
  plt.ylabel('RMSD')
  plt.title('Individual RMSD between CA atoms')
  plt.show()
  print("Mean RMSD:")
  print(str(suma) + ' Å') #el rmsd total según la formula, ahora dando tal como en los otros métodos
  return rmsd_individual
