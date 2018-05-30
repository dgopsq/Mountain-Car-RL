import torch
import numpy as np

from src.custom_net import CustomNet
from src.model import Model
from src.agent import Agent

# Settings random seeds
torch.manual_seed(1); 
np.random.seed(1)

# Setting up bounds
position_bounds = (-1.2, 0.5)
velocity_bounds = (-0.07, 0.07)
actions = np.linspace(-1.0, 1.0, 4)

# Instanced CustomNet
net = CustomNet(2, len(actions))

# Instanced Model
model = Model(
    position_bounds, # Position bounds
    velocity_bounds # Velocity bounds
)

# Instanced Agent
agent = Agent(
    net, # NeuralNetwork class
    model,
    actions, # Actions array (after discretization)
    100, # Max number of episodes
    100000, # Max number of epoches per episode
    0.2 # Greed factor
)

# Getting the result array of len(episodes) length
results = agent.learn()

n_success = [x for x in results if x["state"]["position"] == position_bounds[1]]
print("Number of successful episodes: {0}".format(len(n_success)))
for ep in n_success: 
    print(ep)