from numpy import linspace

from src.custom_net import CustomNet
from src.model import Model
from src.agent import Agent

position_bounds = (-1.2, 0.5)
velocity_bounds = (-0.07, 0.07)
actions = linspace(-1.0, 1.0, 4)

net = CustomNet(2, len(actions))

model = Model(
    position_bounds, # Position bounds
    velocity_bounds # Velocity bounds
)

agent = Agent(
    net, # NeuralNetwork class
    model,
    actions, # Actions array (after discretization)
    100, # Max number of episodes
    100000, # Max number of epoches per episode
    0.2 # Greed factor
)

results = agent.learn()

print("Successful episodes")
print("Number of successful episodes: {0}".format(len(results)))
for ep in results: 
    print(ep)