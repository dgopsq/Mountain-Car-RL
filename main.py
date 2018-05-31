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

episodes = 1500
epoches = 150
greed_factor = 0.1

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
    episodes, # Max number of episodes
    epoches, # Max number of epoches per episode
    greed_factor # Greed factor
)

# Getting the result array of len(episodes) length
results = agent.learn()

# Success episodes
success_results = [x for x in results if x["state"][0] >= position_bounds[1]]

# Writing the log file
log_file = open("model.log", "w+")

log_file.write("Episodes: {0}\n".format(episodes))
log_file.write("Epoches: {0}\n".format(epoches))
log_file.write("Epsilon-Greedy: {0}\n".format(greed_factor))
log_file.write("\n------- {0} successful episodes -------\n\n".format(len(success_results)))

for r in success_results:
    log_file.write(str(r) + "\n")

log_file.close()

# Plotting results
plt.figure(figsize = (15, 7.5))

positions = pd.Series([x["state"][0] for x in results])
position_means = positions.rolling(10).mean()

plt.plot(positions, color = "#00c9b1", alpha = 0.4)
plt.plot(position_means, color = "#005d6c")

plt.ylabel("Position")
plt.xlabel("Episode")

plt.savefig("plot.png")