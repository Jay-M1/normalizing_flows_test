import torch
import torch.distributions as dist

import FrEIA.framework as Ff # Framework for Easily Invertible Architectures
import FrEIA.modules as Fm # collection of invertible operations

import os
import shutil
import yaml

from utils import *

with open('/work/jmustafi/bachelorarbeit/tauFF/nf/configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
DIM = config['dim']
BATCHSIZE = config['batch_size']
N_BLOCKS = config['n_blocks']
OUTPUT_FOLDER = config['output_folder']
HIDDEN_NODES_NN = config['hidden_nodes_nn']
HIDDEN_LAYERS_NN = config['hidden_layers_nn']
N_SAMPLES = config['n_samples']
CLAMP = config['clamp']
NAME = config['name']
EPOCHS = config['n_epochs']
LR = config['learning_rate']
SCHEDULER = config['scheduler']

#######################################

if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

inn = Ff.SequenceINN(DIM)
for _ in range(N_BLOCKS):
    # use RNVP Flow. Its parameters are learned by a MLP
    # clamp is a hyperparameter that influences the regular RNVP transformation to not diverge too much
    inn.append(Fm.RNVPCouplingBlock, subnet_constructor=mlp_constructor, clamp=CLAMP)
    #inn.append(Fm.NICECouplingBlock, subnet_constructor=mlp_constructor)

train(model=inn,
    name=NAME,
    n_epochs=EPOCHS+1,
    lr=LR,
    scheduler=SCHEDULER,
    device=torch.device("gpu" if torch.cuda.is_available() else "cpu"),
    jupyter_nb=False)

print(f"Model is bijective: {is_bijective(inn)}")
