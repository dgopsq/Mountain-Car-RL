import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.policy import Policy
from src.model import Model
from src.agent import Agent

torch.manual_seed(1)
np.random.seed(1)

# Setting up bounds
position_bounds = (-1.2, 0.5)
velocity_bounds = (-0.07, 0.07)
actions = [-1.0, 0.0, 1.0]

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
    3000, # Max number of episodes
    200, # Max number of epoches per episode
    0.3 # Greed factor
)

# Getting the result array of len(episodes) length
results = agent.learn()

# Success episodes
success_results = [x for x in results if x["state"][0] >= position_bounds[1]]
print("Number of successful episodes: {0}".format(len(success_results)))

#for r in results:
#    print(r)

# Plotting results
plt.figure(2, figsize=[10,5])
positions = [x["state"][0] for x in results]
p = pd.Series(positions)
ma = p.rolling(10).mean()
plt.plot(p)
plt.plot(ma)
plt.xlabel('Episode')
plt.ylabel('Position')
plt.savefig('training_result.png')

plt.show()