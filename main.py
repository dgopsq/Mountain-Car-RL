import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.policy import Policy
from src.model import Model
from src.agent import Agent

# Setting up bounds
position_bounds = (-1.2, 0.5)
velocity_bounds = (-0.07, 0.07)
actions = np.linspace(-1.0, 1.0, 3)

# Instanced Policy
policy = Policy(2, len(actions))

# Instanced Model
model = Model(
    position_bounds, # Position bounds
    velocity_bounds # Velocity bounds
)

# Instanced Agent
agent = Agent(
    policy, # NeuralNetwork class
    model,
    actions, # Actions array (after discretization)
    100, # Max number of episodes
    100000, # Max number of epoches per episode
    0.3 # Greed factor
)

# Getting the result array of len(episodes) length
results = agent.learn()

# Success episodes
success_results = [x for x in results if x["state"]["position"] == position_bounds[1]]
print("Number of successful episodes: {0}".format(len(success_results)))

# Plotting results
plt.figure(2, figsize=[10,5])
positions = [x["state"]["position"] for x in results]
p = pd.Series(positions)
ma = p.rolling(10).mean()
plt.plot(p)
plt.plot(ma)
plt.xlabel('Episode')
plt.ylabel('Position')
plt.savefig('training_result.png')

plt.show()