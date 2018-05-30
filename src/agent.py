import torch
import torch.nn as nn
import random as rn
from tqdm import tqdm

class Agent:
    def __init__(self, policy, model, actions, max_episodes, max_epoches, greed_factor, learning_rate = 0.1, discount_factor = 0.99):
        self.policy = policy
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr = learning_rate)

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
        
        results = []

        with tqdm(total = self.max_episodes) as pbar:
            # Episodes cycle
            while current_episode < self.max_episodes:
                pbar.update(1)
                current_episode += 1

                # Initial state
                self.model.reset_state()
                current_epoch = 0
                
                # Epoches cycle
                while current_epoch < self.max_epoches:
                    current_epoch += 1

                    if(self.model.is_success_state()):
                        break

                    next_state = self.execute_epoch()

                # Updating results
                results.append({ 
                    "episode": current_episode, 
                    "epoch": current_epoch, 
                    "state": self.model.get_current_state()
                })
        
        return results

    # Use the neural network to get a new acceleration
    # from the current state
    def execute_epoch(self):
        current_state = self.model.get_current_state()

        # The input state
        current_state_tensor = torch.tensor([
            current_state["position"],
            current_state["velocity"]
        ]).type(torch.FloatTensor)
        
        # The action to take
        current_q_values = self.policy(current_state_tensor)
        best_action = current_q_values.max(0)[1]
        action_index = self._get_next_action(best_action)
        action = self.actions[action_index]

        # The new state
        next_state = self.model.execute_action(action)
        reward = self.model.get_reward()
        
        # The new state max Q action
        next_state_tensor = torch.tensor([
            next_state["position"],
            next_state["velocity"]
        ]).type(torch.FloatTensor)

        # Max Q value in the next state
        next_q_values = self.policy(next_state_tensor).detach()
        next_max_q_value = next_q_values.max(0)[0]

        # Getting the expected Q value
        expected_q_values = current_q_values.detach().clone()
        expected_q_values[action_index] = reward + (next_max_q_value * self.gamma)

        # Learning the network
        loss = self.loss_func(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return next_state
    
    # Return next action's index using epsilon-greedy
    def _get_next_action(self, best_action):
        rnd = rn.random()
        if(rnd < self.epsilon):
            return rn.randint(0, len(self.actions) - 1)
        else:
            return best_action