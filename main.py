import torch
import torch.nn as nn
import random as rn
import numpy as np
from math import cos
from tqdm import tqdm

class CustomNet(nn.Module):
    def __init__(self, n_input, n_output):
        super(CustomNet, self).__init__()
        self.hidden_layer_1 = nn.Linear(n_input, n_output * 2)
        self.output_layer = nn.Linear(n_output * 2, n_output)

    def forward(self, x):
        out = self.hidden_layer_1(x)
        out = self.output_layer(out)
        return out

class Agent:
    def __init__(self, Net, actions, position_bounds, velocity_bounds, learning_rate, discount_factor, greed_factor):
        # 2 inputs, len(actions) * 2 hidden nodes, len(actions) output
        self.net = Net(2, len(actions))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = learning_rate)
        self.loss_func = nn.SmoothL1Loss()

        # Q-function's discount factor
        self.gamma = discount_factor
        # Q-function's greedy factor
        self.epsilon = greed_factor

        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds

        # Discrete actions
        self.actions = actions

        # Max learning cycle
        self.max_episodes = 100
        self.max_epochs = 100000

        # Initial state
        self._reset_state()

    def learn(self):
        current_episode = 0
        current_epoch = 0
        success_list = []

        with tqdm(total = self.max_episodes) as pbar:
            while current_episode != self.max_episodes:
                pbar.update(1)
                current_episode += 1

                if(self.state["position"] == self.position_bounds[1]):
                    success_item = { "episode": current_episode, "epoch": current_epoch }
                    success_list.append(success_item)

                # Initial state
                self._reset_state()
                current_epoch = 0
                
                while current_epoch != self.max_epochs:
                    current_epoch += 1

                    if(self.state["position"] in self.position_bounds):
                        break

                    self.execute_epoch()
        
        return success_list

    # Use the neural network to get a new acceleration
    # from the current state
    def execute_epoch(self):
        # The input state
        state_tensor = torch.tensor([
            self.state["position"],
            self.state["velocity"]
        ])

        # The action to take
        current_q_values = self.net(state_tensor)
        action_index = self._get_next_action(current_q_values)
        action = self.actions[action_index]

        # The new state
        next_state = self._generate_next_state(self.state, action)
        reward = self._get_reward(next_state)
        
        # The new state max Q action
        next_state_tensor = torch.tensor([
            next_state["position"],
            next_state["velocity"]
        ])

        max_q_next_action = self.net(next_state_tensor).detach().max(0)[0]
        expected_q_value = reward + (self.epsilon * max_q_next_action)

        # Learning the network
        loss = self.loss_func(current_q_values[action_index], expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Updating state
        self.state = next_state

    def _get_reward(self, state):
        if(state["position"] == self.position_bounds[1]):
            return 10.0
        else:
            return -1.0
    
    # Return next action's index using epsilon-greedy
    def _get_next_action(self, q_values):
        if(rn.random() < self.epsilon):
            return rn.randint(0, len(self.actions) - 1)
        else:
            return q_values.max(0)[1]

    def _check_bounds(self, value, bounds):
        if(value <= bounds[0]):
            return bounds[0]
        elif(value >= bounds[1]):
            return bounds[1]
        else:
            return value
    
    def _generate_state(self, position, velocity):
        return {
            "position": position,
            "velocity": velocity
        }

    def _reset_state(self):
        self.state = self._generate_state(0.0, 0.0)

    def _calc_velocity(self, state, acceleration):
        if(state["position"] in self.position_bounds):
            return 0

        next_velocity = state["velocity"] + (0.001 * acceleration) - (0.0025 * cos(3 * state["position"]))
        next_velocity = self._check_bounds(next_velocity, self.velocity_bounds)
        return next_velocity

    def _calc_position(self, state, next_velocity):
        next_position = state["position"] + next_velocity
        next_position = self._check_bounds(next_position, self.position_bounds)
        return next_position

    def _generate_next_state(self, state, acceleration):
        next_velocity = self._calc_velocity(state, acceleration)
        next_position = self._calc_position(state, next_velocity)
        return self._generate_state(next_position, next_velocity)

position_bounds = (-1.2, 0.5)
velocity_bounds = (-0.07, 0.07)
actions = np.linspace(-1.0, 1.0, 4)

agent = Agent(
    CustomNet, # NeuralNetwork class
    actions, # Actions array (after discretization)
    position_bounds, # Position bounds
    velocity_bounds, # Velocity bounds
    0.1, # Learning rate
    0.99, # Discount factor
    0.2 # Greed factor
)

results = agent.learn()

print("Successful episodes")
print("Number of successful episodes: {0}".format(len(results)))
for ep in results: 
    print(ep)