import torch
import torch.distributions as dist

import FrEIA.framework as Ff # Framework for Easily Invertible Architectures
import FrEIA.modules as Fm # collection of invertible operations

import os
import shutil
import yaml
import logging
import time

from utils import *

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
log_dir = os.path.join(parent_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

with open('/work/jmustafi/bachelorarbeit/tauFF/nf/configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
DIM = config['dim']
BATCHSIZE = config['batch_size']
N_BLOCKS = config['n_blocks']
PLOT_FOLDER = os.path.join(parent_dir, config['plot_folder'])
HIDDEN_NODES_NN = config['hidden_nodes_nn']
HIDDEN_LAYERS_NN = config['hidden_layers_nn']
N_SAMPLES = config['n_samples']
CLAMP = config['clamp']
NAME = config['name']
EPOCHS = config['n_epochs']
LR = config['learning_rate']
SCHEDULER = config['scheduler']

log_path = os.path.join(log_dir, f'{NAME}.log')
logging.basicConfig(
    filename=log_path,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info("\n")
logging.info("----------------------------------------------")
logging.info("Loaded configuration:")
for key, value in config.items():
    logging.info(f"  {key}: {value}")

if os.path.exists(PLOT_FOLDER):
    shutil.rmtree(PLOT_FOLDER)
os.makedirs(PLOT_FOLDER, exist_ok=True)

#######################################

start_time = time.time()

inn = Ff.SequenceINN(DIM)
for _ in range(N_BLOCKS):
    # use RNVP Flow. Its parameters are learned by a MLP
    # clamp is a hyperparameter that influences the regular RNVP transformation to not diverge too much
    #inn.append(Fm.RNVPCouplingBlock, subnet_constructor=mlp_constructor, clamp=CLAMP)
    inn.append(Fm.NICECouplingBlock, subnet_constructor=mlp_constructor)
    #inn.append(Fm.AffineCouplingOneSided, subnet_constructor=mlp_constructor, clamp=CLAMP)


train(model=inn,
    name=NAME,
    n_epochs=EPOCHS+1,
    lr=LR,
    scheduler=SCHEDULER,
    device=torch.device("gpu" if torch.cuda.is_available() else "cpu"),
    jupyter_nb=False)

logging.info(f"Model is bijective: {is_bijective(inn)}")

end_time = time.time()
elapsed = end_time - start_time
logging.info(f"Total runtime: {elapsed:.2f} seconds")
