import torch
import torch.nn as nn
import random as rn
from tqdm import tqdm

class Agent:
    def __init__(self, net, model, actions, max_episodes, max_epoches, greed_factor, learning_rate = 0.1, discount_factor = 0.99):
        # 2 inputs, len(actions) * 2 hidden nodes, len(actions) output
        self.net = net
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr = learning_rate)
        self.loss_func = nn.SmoothL1Loss()

        # Q-function's discount factor
        self.gamma = discount_factor
        # Q-function's greedy factor
        self.epsilon = greed_factor

        # Model
        self.model = model

        # Discrete actions
        self.actions = actions

        # Max learning cycle
        self.max_episodes = max_episodes
        self.max_epoches = max_epoches

    def learn(self):
        current_episode = 0
        current_epoch = 0
        success_list = []

        with tqdm(total = self.max_episodes) as pbar:
            while current_episode != self.max_episodes:
                pbar.update(1)
                current_episode += 1

                if(self.model.is_success_state()):
                    success_item = { "episode": current_episode, "epoch": current_epoch }
                    success_list.append(success_item)

                # Initial state
                self.model.reset_state()
                current_epoch = 0
                
                while current_epoch != self.max_epoches:
                    current_epoch += 1

                    if(self.model.is_final_state()):
                        break

                    self.execute_epoch()
        
        return success_list

    # Use the neural network to get a new acceleration
    # from the current state
    def execute_epoch(self):
        current_state = self.model.get_current_state()

        # The input state
        current_state_tensor = torch.tensor([
            current_state["position"],
            current_state["velocity"]
        ])

        # The action to take
        current_q_values = self.net(current_state_tensor)
        action_index = self._get_next_action(current_q_values)
        action = self.actions[action_index]

        # The new state
        next_state = self.model.execute_action(action)
        reward = self.model.get_reward()
        
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
    
    # Return next action's index using epsilon-greedy
    def _get_next_action(self, q_values):
        if(rn.random() < self.epsilon):
            return rn.randint(0, len(self.actions) - 1)
        else:
            return q_values.max(0)[1]