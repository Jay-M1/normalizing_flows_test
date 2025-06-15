import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributions as dist

from matplotlib import pyplot as plt
import yaml
import logging

with open('/work/jmustafi/bachelorarbeit/tauFF/nf/configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
DIM = config['dim']
BATCHSIZE = config['batch_size']
N_BLOCKS = config['n_blocks']
PLOT_FOLDER = config['plot_folder']
HIDDEN_NODES_NN = config['hidden_nodes_nn']
HIDDEN_LAYERS_NN = config['hidden_layers_nn']
N_SAMPLES = config['n_samples']
CLAMP = config['clamp']
NAME = config['name']
EPOCHS = config['n_epochs']
LR = config['learning_rate']
SCHEDULER = config['scheduler']
IMPROVEMENT_THRESHOLD = config['improvement_threshold']

def mlp_constructor(input_dim, output_dim, hidden_layers= HIDDEN_LAYERS_NN, hidden_nodes=HIDDEN_NODES_NN):
    """
    Constructs a multi-layer perceptron (MLP) with the specified parameters.
    
    Args:
        input_dim (int): Dimension of the input layer.
        output_dim (int): Dimension of the output layer.
        n_hidden_layers (int): Number of hidden layers in the MLP.
        hidden_nodes (int): Number of nodes in each hidden layer.
        
    Returns:
        nn.Sequential: A sequential model representing the MLP.
    """
    layers = []
    layers.append(nn.Linear(input_dim, hidden_nodes))
    layers.append(nn.ReLU())
    for _ in range(hidden_layers - 1):
        layers.append(nn.Linear(hidden_nodes, hidden_nodes))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_nodes, output_dim))
    return nn.Sequential(*layers)

def target_distribution():
    return dist.MultivariateNormal(5*torch.ones(DIM), torch.eye(DIM))

def target_pdf(z):
    return target_distribution().log_prob(z).exp()

def loss(density, y, log_jacobians):
    if type(log_jacobians) == list:
        log_jacobians = torch.stack(log_jacobians).mean()
    return -log_jacobians.mean() - torch.log(density(y)+1e-9).mean()

def latent_distribution():
    return dist.MultivariateNormal(torch.zeros(DIM), torch.eye(DIM))

def train(model,
        name=None,
        n_epochs=1001,
        lr=1e-3,
        scheduler=None,
        device=torch.device("cpu"),
        jupyter_nb=True,):
    
    pz = latent_distribution()

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduler is not None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, scheduler)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Draw sample from latent distribution
        z = pz.sample((BATCHSIZE,))
        z = z.to(device)

        # learn
        y, log_jacobians = model(z)
        loss_value = loss(target_pdf, y, log_jacobians)
        loss_value.backward()
        optimizer.step()

        # Early stopping: exit if loss doesn't improve for 20 epochs
        if epoch == 0:
            best_loss = loss_value.item()
            epochs_no_improve = 0
        else:
            if loss_value.item() < best_loss*(1-IMPROVEMENT_THRESHOLD):  
                best_loss = loss_value.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= 20:
                logging.info(f"Early stopping at epoch {epoch} (no improvement for at least {IMPROVEMENT_THRESHOLD*100}% for 20 epochs).")
                break

        if scheduler is not None:
            scheduler.step()
        
        if (epoch % int(n_epochs/10.) == 0):
            logging.info(f"Epoch {epoch}, Loss: {loss_value.item():.4f}")

            # plot
            z_samples = pz.sample((N_SAMPLES,)).to(device)
            y_samples, _ = model(z_samples)
            y_samples = y_samples.detach().cpu().numpy()

            plt.figure(figsize=(6, 6))
            if DIM >= 2:
                plt.scatter(y_samples[:, 0], y_samples[:, 1], s=1, alpha=0.5)
            else:
                plt.hist(y_samples[:, 0], bins=70, alpha=0.7)
                plt.xlim(-2, 10)
            plt.title(f"{name}\nSamples at epoch {epoch}\nSample mean: {y_samples[:, 0].mean():.2f}")
            #plt.xlabel("x1")
            #plt.ylabel("x2")
            if jupyter_nb:
                plt.show()
            else:
                plt.savefig(f"{PLOT_FOLDER}/epoch_{epoch}.png")

def is_bijective(model):
    """
    Checks if the model is bijective by sampling from the latent space and reversing the flow.
    """
    pz = latent_distribution()
    z = pz.sample((int(N_SAMPLES), ))

    model.cpu()
    y, _ = model(z)
    z_rev, _ = model(y, rev=True)
    return bool(torch.max(torch.abs(z_rev - z)) < 1e-5)
