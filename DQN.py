# DQN
## Imports
"""

import matplotlib.pyplot as plt
import imageio
import os
import gym
from gym import spaces
# error: random_sample means conflict between numpy and py
import random
import numpy as np
from collections import deque
import copy
import pickle

# no IPython magic to ensure Python compatibility.
import matplotlib.mlab as mlab
import matplotlib.animation
from IPython.display import HTML
from matplotlib import rc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""## Environment"""

class HomeostaticEnv(gym.Env):
    """
    Custom Environment that follows gym interface.

    Parameters:
    - AgentStats: Class for managing stats of agents
    - loc: Initial location of the agent
    - n_stats: Number of features (one for each stat)
    - size: Size of the gridworld (grid_size x grid_size)
    - fov: Number of cells the agent can see in each direction
    - view_size: Height and width of the view square
    - border: Colour of border and of agent on grid
    - bounds: Range of means for resource patches
    - history: History for visualization
    - resource_percentile: Resources affect stat above this
    """

    def __init__(self,
                 AgentStats,
                 n_stats,
                 grid_size=10,
                 stationary=True,
                 field_of_view=1,
                 loc=[5, 5],
                 n_actions=4,
                 bounds=3,
                 radius=3.5,
                 offset=0.5,
                 variance=1,
                 resource_percentile=95
                 ):
        super(HomeostaticEnv, self).__init__()

        # Agent parameters
        # 2D location [x, y] where x = rows and y = columns
        # 0 == the row on the top or the column on the left
        self.loc = loc
        self.n_actions = n_actions

        # Stats parameters
        self.AgentStats = AgentStats
        self.n_stats = n_stats

        # Grid world parameters
        self.stationary = stationary
        self.size = grid_size
        self.fov = field_of_view
        self.view_size = 2 * self.fov + 1
        self.border = -0.02
        self.n_statedims = self.n_stats * self.view_size ** 2 + self.n_stats # Current stats + all stats in view

        # Resource parameters
        self.bounds = bounds
        self.radius = radius
        self.offset = offset
        self.variance = variance
        self.resource_percentile = resource_percentile

        # History
        #self.history = []
        self.save = False
        self.location_history = []
        self.stats_history = []
        self.heat_map = np.zeros((self.size, self.size))

        # Initialize the environment --Reset
        self.grid = np.zeros((self.n_stats, self.size, self.size))  # empty world grid
        self.reset_stats()
        self.reset_grid()
        self.reset()

    ######## Reset
    def reset(self):
        if not self.stationary:
            self.reset_grid()
        self.time_step = 0  # resets step timer
        self.done = False
        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
            self.loc[1] - self.fov: self.loc[1] + self.fov + 1]  # gets initial agent view
        return self.get_state()

    def reset_stats(self):
        """Initialize stats of the agent."""
        self.stats = []
        for i in range(self.n_stats):
            self.stats.append(np.random.uniform(self.AgentStats.initial_stats[0], self.AgentStats.initial_stats[1]))

    def reset_grid(self):
        """Reset both thresholds and distribution of resources."""
        self.thresholds = []
        if not self.stationary:
            self.offset += np.random.uniform(0, 2 * np.pi)
        self.grid = get_n_resources(grid_size=self.size, bounds=self.bounds, resources=self.n_stats,
                                    radius=self.radius, offset=self.offset, var=self.variance)
        for stat in range(self.n_stats):
            # Returns a single stat value: the 95th percentile (resource_percentile=95)(high) in the whole grid world
            self.thresholds.append(np.percentile(self.grid[stat, :, :], self.resource_percentile))

        # Set all stats on borders to  specific value (i.e., self.border)
        self.grid[:, [0, -1], :] = self.grid[:, :, [0, -1]] = self.border

        self.original_grid = copy.copy(self.grid)

    ####### Step
    def step(self, action, ep_length):
        """Update locations, stats, and rewards."""
        self.time_step += 1

        # Check if this episode terminates
        if self.time_step == ep_length:
            self.done = True

        # Move
        if action == 0 and self.loc[0] < self.size - self.fov - 1:
            self.loc[0] += 1
        elif action == 1 and self.loc[0] > self.fov:
            self.loc[0] -= 1
        elif action == 2 and self.loc[1] < self.size - self.fov - 1:
            self.loc[1] += 1
        elif action == 3 and self.loc[1] > self.fov:
            self.loc[1] -= 1

        # Get reward
        reward, module_rewards = self.step_stats()

        if self.AgentStats.reward_clip is not None:
            reward = np.clip(reward, -self.AgentStats.reward_clip,
                            self.AgentStats.reward_clip)  # reward clipping

        if self.AgentStats.squeeze_rewards:
            reward = np.tanh(reward)
            module_rewards = np.tanh(module_rewards)

        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
                    self.loc[1] - self.fov: self.loc[1] + self.fov + 1]

        # Store history
        if self.save:
            self.stats_history.append(copy.deepcopy(self.stats))
            #self.history.append((self.grid_with_agent(), copy.deepcopy(self.stats)))
            self.location_history.append(copy.copy(self.loc))
        self.heat_map[self.loc[0], self.loc[1]] += 1
        # What is appended:
        # Current position in the grid (4*10*10), and current stats (4)
        # Shape of history: num_frame * 2 *

        # Return step information
        return self.get_state(), reward, self.done, module_rewards

    def step_stats(self):
        """
        Decrease stats over time;
        Increase stats when resources are above thresholds;
        Return rewards.

        Returns:
        Reward in the current time step.
        """
        self.old_stats = copy.deepcopy(self.stats)

        for i in range(self.n_stats):
            # All stats decrease over time
            self.stats[i] -= self.AgentStats.stat_decrease_per_step
            # Stats only increase when resources are above thresholds
            if self.grid[i, self.loc[0], self.loc[1]] > self.thresholds[i]:
                self.stats[i] += self.grid[i, self.loc[0], self.loc[1]]

        #avg = self.AgentStats.avg_stats(self.stats)

        if self.AgentStats.reward_type == 'HRRL':
            reward = self.AgentStats.HRRL_reward(self.old_stats, self.stats)

        if self.AgentStats.module_reward_type == 'HRRL':
            module_rewards = self.AgentStats.separate_HRRL_rewards(self.old_stats, self.stats)

        return reward, module_rewards

    def get_state(self):
        """
        Returns:
        A 1D array of shape: n_stats (current stats) + n_stats * view ** 2 (all stats in view)
        """
        return torch.cat((torch.tensor(self.stats).float(), torch.tensor(self.view.flatten()).float())).float()

    #################################
    # Visualization
    def grid_with_agent(self):
        """
        Mark the agent's current position in the grid by setting it to a specified value (self.border).
        (Mainly for visualization)
        """
        temp = copy.copy(self.grid) # Shallow copy, won't affect grid
        temp[:, self.loc[0], self.loc[1]] = self.border

        return temp

    def move_location(self, loc):
        """
        (Mainly for visualization)
        """
        self.loc = loc
        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
                    self.loc[1] - self.fov: self.loc[1] + self.fov + 1]

    def render(self, mode='human'):
        """
        (Mainly for visualization)
        """
        tits = [f'stat {i + 1}' for i in range(self.n_stats)]
        for i in range(self.n_stats):
            plt.subplot(100 + 10 * self.n_stats + 1 + i)
            plt.title(tits[i])
            plt.imshow(self.grid_with_agent()[i])
            plt.xticks([])
            plt.yticks([])
        plt.figure()
        plt.imshow(self.heat_map)
        plt.show()

def get_n_resources(grid_size=20, bounds=4, resources=3, radius=2, offset=0, var=1):
    # The distribution on the variables X, Y packed into pos.
    z = np.zeros((resources, grid_size, grid_size))

    sigma = [[var,0],[0,var]] # covariance matrix for the Gaussian distributions

    points = get_spaced_means(resources, radius, offset)
    for i in range(resources):
        z[i,:,:] = multivariate_gaussian(grid_size = grid_size, bounds = bounds,
                                        m1 = points[i][0], m2 = points[i][1], Sigma = sigma)
    return z

def get_spaced_means(modes, radius, offset):
    radians_between_each_point = 2*np.pi/modes
    list_of_points = []
    for p in range(0, modes):
        node = p + offset
        list_of_points.append( (radius*np.cos(node*radians_between_each_point),radius*np.sin(node*radians_between_each_point)) )
    return list_of_points

def multivariate_gaussian(grid_size = 20, bounds = 4, height = 1, m1 = None, m2 = None, Sigma = None):
    """
    Return the multivariate Gaussian distribution on array pos.

    Parameters:
    - bounds: Range for the Gaussian distribution.
    - height: Maximum height (scaling factor) of the Gaussian distribution.
    - m1 & m2: Means for the Gaussian distribution in the X and Y directions.
    - Sigma: Covariance matrix for the Gaussian distribution.
    """

    # Generate grid_size evenly spaced points between -bounds and bounds
    X = np.linspace(-bounds, bounds, grid_size) # shape: grid_size
    Y = np.linspace(-bounds, bounds, grid_size) # shape: grid_size
    X, Y = np.meshgrid(X, Y)                    # X / Y shapes: [grid_size, grid_size]
    # Each point in the grid = (X[i,j], Y[i,j]) -- a pair of X & Y

    # Mean vector
    mu = np.array([m1, m2])

    height = np.random.uniform(1, height)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,)) # shape: [grid_size, grid_size, 2]
    # Put X and Y into pos
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    # Every smallest element in pos has a length of 2: [xn, yn]

    n = mu.shape[0]
    # Determinant of covariance matrix:
    Sigma_det = np.linalg.det(Sigma) # If variance = 2, here Sigma_det = variance**2 = 4
    # Inverse of covariance matrix:
    Sigma_inv = np.linalg.inv(Sigma)
    # Normalization Factor: make sure area under curve = 1
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    # Calculate the Mahalanobis distance for each point in the grid
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return height*norm(np.exp(-fac / 2) / N)

def norm(z):
    """
    Normalize an array z such that the sum of its elements equals 1
    """
    return z/z.sum()

"""## Stats"""

class AgentStats:
    def __init__(self,
                 n_stats,
                 set_point=5,
                 initial_stats=[0.5, 0.5],
                 stat_decrease_per_step=0.01,
                 squeeze_rewards=True,
                 reward_type='HRRL',
                 module_reward_type='HRRL',
                 reward_scaling=1,
                 reward_clip=500,
                 mod_reward_scaling=1,
                 mod_reward_bias=0,
                 pq=[4,2]
                 ):

        self.n_stats = n_stats
        self.set_point = set_point
        self.initial_stats = initial_stats
        self.stat_decrease_per_step = stat_decrease_per_step

        # Reward relevant
        self.squeeze_rewards = squeeze_rewards
        self.reward_scaling = reward_scaling
        self.reward_clip = reward_clip
        self.mod_reward_scaling = mod_reward_scaling
        self.reward_type = reward_type
        self.module_reward_type = module_reward_type
        self.mod_reward_bias = mod_reward_bias
        self.p, self.q = pq[0], pq[1]  # homeostatic RL exponents for reward function

    def avg_stats(self, stats):
        return sum(stats) / self.n_stats

    def HRRL_reward(self, old_stats, stats):
        return self.reward_scaling * (self.get_cost_surface(old_stats) - self.get_cost_surface(stats))

    def separate_HRRL_rewards(self, old_stats, stats):
        return [self.mod_reward_scaling * ((np.abs(self.set_point - old_stat) ** self.p) ** (1 / self.q) - (
                    np.abs(self.set_point - new_stat) ** self.p) ** (1 / self.q)) - self.mod_reward_bias for
                old_stat, new_stat in zip(old_stats, stats)]

    def get_cost_surface(self, stats):
        return sum([np.abs(self.set_point - stat) ** self.p for stat in stats]) ** (1 / self.q)

"""## Replay buffer"""

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        Initialize the replay buffer.

        Parameters:
        - buffer_size: The maximum number of experiences that the buffer can hold.
        - batch_size: The number of experiences to sample for each training step.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        # deque (double-ended queue): append the newest and pop the oldest
        # maxlen: limits the maximum size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Parameters:
        - state: The current state observed by the agent.
        - action: The action taken by the agent.
        - reward: The reward received after taking the action.
        - next_state: The next state observed after taking the action.
        - done: A boolean indicating whether the episode has ended.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        """
        Sample a batch of experiences from the buffer.

        Returns:
        A tuple containing batches of states, actions, rewards, next states, and done flags.
        """
        batch = random.sample(self.buffer, self.batch_size) # avoid conflict with np.random
        states, actions, rewards, next_states, dones = zip(*batch) # unpack
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns:
        The number of experiences currently stored in the buffer.
        """
        return len(self.buffer)

"""## Network"""

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, blind=True, hidden_size=1024):
        """
        Initialize the Q-Network.

        Parameters:
        - state_size: The size of the input state space.
        - action_size: The size of the output action space.
        - hidden_size: The number of neurons in the hidden layers (default is 64).
        """
        super(QNetwork, self).__init__()

        self.blind = blind
        # Define the layers
        self.fc1 = nn.Linear(state_size, hidden_size) # Hidden (input)
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Hidden
        self.fc3 = nn.Linear(hidden_size, action_size) # Output layer

        if self.blind:
            self.mask = nn.Parameter(0.1 + torch.zeros(1, state_size)) # This is the mask (it is just an array initialized with all 0.1s)

    def forward(self, state):
        """
        Forward pass through the network.

        Parameters:
        - state: The input state.

        Returns:
        - Q-values for each action.
        """
        if self.blind:
            state = state * self.mask

        # Pass the input state through the first hidden layer with ReLU activation
        x = F.relu(self.fc1(state))
        # Pass the result through the second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Pass the result through the output layer to get Q-values for each action
        return self.fc3(x)

"""## E-greedy policy"""

class EpsilonGreedyPolicy:
    def __init__(self, decay_rate, decision_process='gmQ', epsilon_start=1, epsilon_end=0.01):
        """
        Initialize the epsilon-greedy policy.

        Parameters:
        - epsilon_start: Initial value of epsilon (high exploration).
        - epsilon_end: Final value of epsilon (low exploration).
        - decay_rate: Rate at which epsilon decays.
        """
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.decision_process = decision_process

    def select_action(self, q_network, state, action_size):
        """
        Select an action using the epsilon-greedy policy.

        Parameters:
        - q_network: The Q-network used to predict Q-values.
        - state: The current state of the environment.
        - action_size: The number of possible actions.

        Returns:
        - The selected action.
        """
        # With probability epsilon, choose a random action (exploration)
        if random.random() < self.epsilon: # Generates a random float number between 0 and 1
            selected_action = random.randint(0, action_size - 1) # 0 & action_size - 1 are inclusive
        else:
            if self.decision_process == "None":
                # Otherwise, choose the action with the highest Q-value (exploitation)
                # PyTorch expects inputs with a batch dimension, even if the batch size is 1
                state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension
                # Gradients are not needed for inference (exploitation)
                # No parameter updates
                with torch.no_grad(): # Disable gradient calculation
                    # PyTorch: calling an instance of a torch.nn.Module subclass with input
                    # data automatically calls the forward method
                    q_values = q_network(state) # q_network: an instance of the QNetwork class
                #print(q_values)
                selected_action = q_values.argmax().item() # Returns the index of the maximum Q-value
            if self.decision_process == "gmQ":
                prepared_state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension
                with torch.no_grad():  # Disable gradient calculation
                    ModuleQValues = [q_network[i](prepared_state).detach() for i in range(len(q_network))]
                finalQValues = sum(ModuleQValues)
                selected_action = finalQValues.argmax().item()

        return selected_action

    def decay_epsilon(self):
        """
        Decay the value of epsilon.
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) * self.decay_rate
            )

        return self.epsilon

"""## Monolithic DQN agent"""

class DQNAgent:
    def __init__(self, env, policy, action_size, learning_rate, gamma, lambd, blind = False, hidden_size=64):
        """
        Initialize the DQN agent.

        Parameters:
        - policy: Policy used to select actions (e.g., epsilon-greedy)
        - action_size: Dimension of the action space.
        - hidden_size: Number of neurons in the hidden layers.
        - learning_rate: Learning rate for the optimizer.
        """
        self.env = env
        self.is_modular = False
        self.n_statedims = self.env.n_statedims
        self.action_size = action_size
        self.hidden_size = hidden_size
        # Determine the importance of future rewards (0, 1)
        self.gamma = gamma  # Discount factor

        self.blind = blind
        self.lambd = lambd

        # Initialize Q-network and target network
        self.q_network = QNetwork(self.n_statedims, self.action_size, self.blind, self.hidden_size).to(device)
        self.target_network = QNetwork(self.n_statedims, self.action_size, self.blind, self.hidden_size).to(device) # Copy Q-network
        # Copy parameters from Q-network to target network as to start with same parameters
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Evaluation mode: This will not affect parameter copying
        self.target_network.eval()

        # Initialize Adam optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        """
        Select an action using the policy (e.g., epsilon-greedy).

        Parameters:
        - state: Current state.

        Returns:
        - action: Selected action.
        """
        return policy.select_action(self.q_network, state, self.action_size)

    def update_q_network(self, loss):
        """
        Update the model by gradient descent.
        """

        # Perform backpropagation and update the Q-network
        self.optimizer.zero_grad() # Clears the old gradients from the last step
        loss.backward() # Computes the gradient of the loss
        #torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step() # Updates parameters based on the computed gradients

    def update_target_network(self):
        """
        Update the target network to match the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """
        Decay the epsilon value of the epsilon-greedy policy.

        Returns:
        - epsilon: The value of epsilon in the next time step
        """
        epsilon = policy.decay_epsilon()
        return epsilon

    def compute_dqn_loss(self, replay_buffer):
        """
        Sample from the replay buffer;
        Calculate Q values (i.e., current, next, target Q values);
        Return DQN loss.

        Returns:
        - loss: Computed dqn loss
        """
        # Sample from the replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample()

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device) # Interger type: to be used for "gather" function
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Compute current Q-values using the Q-network
        # self.q_network(states) returns: [batch_size, 1, action_size]
        # actions.unsqueeze(1) is used to extract the Q-values for specific actions
        states = states.squeeze(1)  # Remove the extra dimension if it's always 1
        #print(f"Shape of states: {states.shape}")

        curr_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        #print(f"Shape of curr_q_values: {curr_q_values.shape}")

        # Use torch.no_grad() because we do not need gradients for these values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).gather(  # Double DQN
                1, self.q_network(next_states).argmax(dim=1, keepdim=True)
            ).squeeze(1).detach()
        #print(f"Shape of next_q_values: {next_q_values.shape}")

        # Compute target Q-values
        # Return the maximum Q-value for each next state across all possible actions
        #next_q_values = self.target_network(next_states).max(1)[0] # Shape: [batch_size]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        #print(f"Shape of target_q_values: {target_q_values.shape}")

        # Calculate dqn loss
        if self.blind:
            L1_loss = self.lambd * torch.sum(torch.abs(self.q_network.mask)) # this is the 'penalty' for having numbers other than 0
            loss = F.smooth_l1_loss(curr_q_values, target_q_values) + L1_loss
        else:
            loss = F.smooth_l1_loss(curr_q_values, target_q_values)

        return loss

"""## Modular DQN agent"""

class Modular_DQNAgent:
    def __init__(self, env, policy, action_size, numModules, learning_rate, gamma, lambd, blind=False, hidden_size=64):
        """
        Initialize the DQN agent.

        Parameters:
        - action_size: Dimension of the action space.
        - hidden_size: Number of neurons in the hidden layers.
        - learning_rate: Learning rate for the optimizer.
        """
        self.env = env
        self.is_modular = True
        self.n_statedims = self.env.n_statedims
        self.numModules = numModules
        self.action_size = action_size
        self.hidden_size = hidden_size
        # Determine the importance of future rewards T(0, 1)
        self.gamma = gamma  # Discount factor

        self.blind = blind
        self.lambd = lambd

        # Initialize Q-network and target network
        self.modules = [
            QNetwork(self.n_statedims, self.action_size, self.blind, self.hidden_size).to(device)
            for i in range(self.numModules)
            ]
        self.targets = [
            QNetwork(self.n_statedims, self.action_size, self.blind, self.hidden_size).to(device)
            for i in range(self.numModules)
            ] # Copy Q-network

        for i in range(self.numModules):
            # Copy parameters from Q-network to target network as to start with same parameters
            self.targets[i].load_state_dict(self.modules[i].state_dict())
            # Evaluation mode: This will not affect parameter copying
            self.targets[i].eval()

        # Initialize Adam optimizer
        self.optimizers = [optim.Adam(modules.parameters(), lr=learning_rate) for modules in self.modules]

    def select_action(self, state):
        """
        Select an action using the policy (e.g., epsilon-greedy).

        Parameters:
        - state: Current state.

        Returns:
        - action: Selected action.
        """
        return policy.select_action(self.modules, state, self.action_size)

    def update_q_network(self, losses):
        """
        Update the model by gradient descent.
        """
        for i in range(self.numModules):
            # Perform backpropagation and update the Q-network
            self.optimizers[i].zero_grad() # Clears the old gradients from the last step
            losses[i].backward() # Computes the gradient of the loss
            #torch.nn.utils.clip_grad_norm_(self.modules[i].parameters(), 1.0)  # Gradient clipping
            self.optimizers[i].step() # Updates parameters based on the computed gradients

    def update_target_network(self):
        """
        Update the target network to match the Q-network.
        """
        for i in range(self.numModules):
            self.targets[i].load_state_dict(self.modules[i].state_dict())

    def decay_epsilon(self):
        """
        Decay the epsilon value of the epsilon-greedy policy.
        """
        epsilon = policy.decay_epsilon()
        return epsilon

    def compute_dqn_loss(self, replay_buffer):
        """
        Return dqn losses in a for loop for each module.
        """
        losses = []

        for i in range(self.numModules):
            # Sample from the replay buffer
            states, actions, rewards, next_states, dones = replay_buffer[i].sample()

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device) # Interger type: to be used for "gather" function
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            # Compute current Q-values using the Q-network
            # self.q_network(states) returns: [batch_size, 1, action_size]
            # actions.unsqueeze(1) is used to extract the Q-values for specific actions
            states = states.squeeze(1)  # Remove the extra dimension if it's always 1
            #print(states.shape)

            curr_q_values = self.modules[i](states).gather(1, actions.unsqueeze(1)).squeeze(1)
            #print(f"Shape of curr_q_values: {curr_q_values.shape}")

            # Double DQN
            with torch.no_grad():
                next_q_values = self.targets[i](next_states).gather(
                    1, self.modules[i](next_states).argmax(dim=1, keepdim=True)
                ).squeeze(1).detach()
            #print(f"Shape of next_q_values: {next_q_values.shape}")

            # Compute target Q-values
            # Return the maximum Q-value for each next state across all possible actions
            #next_q_values = self.target_network(next_states).max(1)[0] # Shape: [batch_size]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            #print(f"Shape of target_q_values: {target_q_values.shape}")

            # Calculate dqn loss
            if self.blind:
                mask = self.modules[i].mask
                L1_loss = self.lambd * torch.sum(torch.abs(mask))
                loss = F.smooth_l1_loss(curr_q_values, target_q_values) + L1_loss
                losses.append(loss)
            else:
                loss = F.smooth_l1_loss(curr_q_values, target_q_values)
                losses.append(loss)

        return losses

    def softmax(self,x):
        return np.exp(x)/sum(np.exp(x))

"""## Trainer"""

class DQNTrainer:
    """
    - Train monolithic or modular agents.
    - Visualize
    """
    def __init__(self, env, agent, replay_buffer, target_update_freq=200, plotting=True, num_frames=30000, ep_length=100000):
        """
        Initialize the DQN trainer.

        Parameters:
        - env: The environment in which the agent operates.
        - agent: The DQNAgent instance.
        - replay_buffer: The ReplayBuffer instance.
        - target_update_freq: Frequency (i.e., timesteps) of updating the target network.
        - plotting: Whether to plot the training process (for epsilons, stats, and v-maps).
        - num_frames: Number of frames to train.
        """
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.num_frames = num_frames
        self.target_update_freq = target_update_freq
        self.ep_length = ep_length

        # Visualization
        self.store_value_maps = False
        self.plotting_interval = 100
        self.plotting = plotting
        self.value_map_history = []

    def train(self):
        """
        Train the DQN agent.
        """
        state = self.env.reset().unsqueeze(0)
        done = False
        update = False
        update_cnt = 0
        score = 0
        self.cumulative_deviation = 0
        self.maximum_deviation = 0
        # The following lists are used for visualization only
        losses = []
        scores = []
        rewards = []
        self.mod_rewards = []
        self.stats = []

        for frame in range(1, self.num_frames + 1):

            if frame % 5000 == 0:
                print(f"Training: time step {frame}")

            # Get action, state and reward
            #print(state)
            action = self.agent.select_action(state)
            next_state, reward, done, sep_rewards = self.env.step(action, self.ep_length)
            #print(sep_rewards)

            score += reward
            #rewards.append(reward)
            self.mod_rewards.append(sep_rewards)
            self.stats.append(copy.deepcopy(self.env.stats))

            # Store action, state and reward
            if not self.agent.is_modular: # if monolithic agent
                self.replay_buffer.add(state, action, reward, next_state, done)
                if len(self.replay_buffer) >= self.replay_buffer.batch_size:
                    update = True
            else: # if modular agent
                tem = 0
                for i in range(self.agent.numModules):
                    buffer = self.replay_buffer[i]
                    if len(buffer) >= buffer.batch_size:
                        tem += 1
                    buffer.add(state, action, sep_rewards[i], next_state, done)
                if tem == len(self.replay_buffer):
                    update = True

            state = next_state.unsqueeze(0)

            # Update networks
            if update:
                # Update Q network
                loss = self.agent.compute_dqn_loss(self.replay_buffer)
                self.agent.update_q_network(loss)
                #losses.append(loss)
                update_cnt += 1
                # Update target network
                if update_cnt % self.target_update_freq == 0:
                    self.agent.update_target_network()

            # Calculate deviation
            if frame >= 15000:
                self.cumulative_deviation += sum([abs(stat - 5) for stat in self.env.stats])
            if frame == self.num_frames:
                self.cumulative_deviation = self.cumulative_deviation / (self.num_frames - 15000)
                print(f"Cumulative deviation is: {self.cumulative_deviation}")
            for stat in self.env.stats:
                if abs(stat) > self.maximum_deviation: self.maximum_deviation = abs(stat)

            # Decay epsilon after each episode
            epsilon = self.agent.decay_epsilon()

            # If episode ends
            if done:
                state = self.env.reset().unsqueeze(0)
                scores.append(score)
                score = 0

            # Visualization
            if self.store_value_maps:
                if not self.agent.is_modular: # if monolithic agent
                    self.mono_get_v_map(display=False)
                else:
                    self.mod_get_v_map(display=False)

            if frame % self.plotting_interval == 0 and self.plotting:
                self.plot(frame, self.stats)
                self.env.render()
                plt.show()


    ###################################################
    # Visualization
    def mono_get_v_map(self, stats_force=None, display=False):
        v_map = np.zeros((self.env.size, self.env.size))

        move_back_loc = copy.copy(self.env.loc)

        ## will take environment and agent as parameters
        for i in range(1, self.env.size - 1):
            for j in range(1, self.env.size - 1):
                self.env.move_location([i, j]) # "move" to the new location and get view
                state = self.env.get_state() # current stats + all stats in view
                #print(state)
                #print(f"location: [{i}, {j}]: {state}")
                if stats_force is not None: state[:self.env.n_stats] = stats_force
                with torch.no_grad():
                    q_vals = self.agent.q_network(torch.FloatTensor(state).to(device)).detach().cpu().numpy()
                    #print(q_vals)
                v_map[i, j] = np.max(q_vals)

        self.env.move_location(move_back_loc)

        if display:
            fig, ax = plt.subplots(figsize=(8, 8))
            self.display_grid_with_text(v_map[1:-1, 1:-1], ax)
            plt.show()
        if not display:
            self.value_map_history.append(v_map[1:-1, 1:-1])

    def mod_get_v_map(self, stats_force=None, display=False):
        v_map = np.empty((self.agent.numModules, self.env.size, self.env.size))
        move_back_loc = copy.copy(self.env.loc)
        ## will take environment and agent as parameters
        for module in range(self.agent.numModules):
            for i in range(1, self.env.size - 1):
                for j in range(1, self.env.size - 1):

                    self.env.move_location([i, j])
                    state = self.env.get_state()
                    if stats_force is not None: state[:self.env.n_stats] = stats_force
                    with torch.no_grad():
                        q_vals = self.agent.modules[module](torch.FloatTensor(state).to(device)).detach().cpu().numpy()
                    v_map[module, i, j] = np.max(q_vals)

        if display:
            fig, axes = plt.subplots(1, self.agent.numModules, figsize=(20, 20))
            if self.agent.numModules > 1:
                for i, ax in enumerate(axes):
                    self.display_grid_with_text(v_map[i, 1:-1, 1:-1], ax)
                    ax.set_title(f'Module {i + 1}')
            else:
                self.display_grid_with_text(v_map[0, 1:-1, 1:-1], axes)
            plt.show()

        if not display:
            self.value_map_history.append(v_map[:, 1:-1, 1:-1])

        self.env.move_location(move_back_loc)

    def plot(
            self,
            frame,
            stats
            ):
        """
        Plot the training progresses.
        """
        plt.figure(figsize=(8, 5))
        #plt.subplot(131)
        #plt.title('rewards')
        #plt.plot(self.mod_rewards)

        #plt.subplot(132)
        #plt.title('epsilons')
        #plt.plot(epsilons)

        #plt.subplot(132)
        plt.title('stats')
        plt.plot(stats)

        plt.ylim([-10, 10])
        plt.legend([f'stat {i + 1}: {np.round(self.env.stats[i], 2)}' for i in range(self.env.n_stats)])
        plt.xlabel('Time step')
        plt.ylabel('Stat levels')
        plt.show()

    def display_grid_with_text(self, grid, ax):
        grid = np.round(grid, 2)
        ax.imshow(grid)
        for (j, i), label in np.ndenumerate(grid):
            ax.text(i, j, label, ha='center', va='center', fontsize=10, fontweight='bold', color='r')
            ax.set_yticks([])
            ax.set_xticks([])

"""## Run

### Monolithic
"""

import copy
# Initialize
for i in range(5):
    print(f"Running monolithic model {i}...")
    cum_deviation = []
    mono_stat_list = []
    n_stats = 4
    n_frames = 30000

    stat_manager = AgentStats(n_stats,
                    set_point=5,
                    initial_stats=[0.5, 0.5],
                    stat_decrease_per_step=0.01,
                    squeeze_rewards=True,
                    reward_type='HRRL',
                    module_reward_type='HRRL',
                    reward_scaling=1,
                    reward_clip=500,
                    mod_reward_scaling=1,
                    mod_reward_bias=0,
                    pq=[4,2])
    env_mono = HomeostaticEnv(stat_manager,
                            n_stats=n_stats,
                            grid_size=10,
                            stationary=True,
                            field_of_view=1,
                            loc=[5, 5],
                            n_actions=4,
                            bounds=3,
                            radius=3.5,
                            offset=0.5,
                            variance=1,
                            resource_percentile=95)
    replay_buffer = ReplayBuffer(buffer_size=100000, batch_size=512)
    policy = EpsilonGreedyPolicy(decay_rate=0.0005,
                                decision_process="None",
                                epsilon_start=1,
                                epsilon_end=0.01)
    mono_agent = DQNAgent(env_mono,
                        policy,
                        action_size=env_mono.n_actions,
                        learning_rate=0.001,
                        gamma=0.5,
                        lambd=0.0001,
                        blind=True,
                        hidden_size=512)
    trainer_mono = DQNTrainer(env_mono,
                        mono_agent,
                        replay_buffer,
                        target_update_freq=200,
                        plotting=False,
                        num_frames=n_frames,
                        ep_length=100000)

    # Start training
    env_mono.save = False
    trainer_mono.store_value_maps = False
    trainer_mono.train()
    cum_deviation.append(trainer_mono.cumulative_deviation)
    mono_stat_list.append(trainer_mono.stats)

with open(f'mono_deviation_list.pkl', 'wb') as file:
    pickle.dump(cum_deviation, file)
with open(f'mono_stat_list.pkl', 'wb') as file:
    pickle.dump(mono_stat_list, file)

"""### Modular"""

import copy
# Initialize
for i in range(5):
    print(f"Running modular model {i}...")
    mod_cum_deviation = []
    mod_stat_list = []
    n_stats = 4
    n_frames = 30000
    stat_manager = AgentStats(n_stats,
                    set_point=5,
                    initial_stats=[0.5, 0.5],
                    stat_decrease_per_step=0.01,
                    squeeze_rewards=True,
                    reward_type='HRRL',
                    module_reward_type='HRRL',
                    reward_scaling=1,
                    reward_clip=500,
                    mod_reward_scaling=1,
                    mod_reward_bias=0,
                    pq=[4,2])
    env_mod = HomeostaticEnv(stat_manager,
                            n_stats=n_stats,
                            grid_size=10,
                            stationary=True,
                            field_of_view=1,
                            loc=[5, 5],
                            n_actions=4,
                            bounds=3,
                            radius=3.5,
                            offset=0.5,
                            variance=1,
                            resource_percentile=95)
    replay_buffer = []
    for i in range(n_stats):
        replay_buffer.append(ReplayBuffer(buffer_size=100000, batch_size=512))
    policy = EpsilonGreedyPolicy(decay_rate=0.0005,
                                decision_process="gmQ",
                                epsilon_start=1,
                                epsilon_end=0.01)
    mod_agent = Modular_DQNAgent(env_mod,
                                policy,
                                action_size=env_mod.n_actions,
                                numModules=n_stats,
                                learning_rate=0.001,
                                gamma=0.5,
                                lambd=0.0001,
                                blind=True,
                                hidden_size=512)
    trainer_mod = DQNTrainer(env_mod,
                        mod_agent,
                        replay_buffer,
                        target_update_freq=200,
                        plotting=False,
                        num_frames=n_frames,
                        ep_length=100000)

    # Start training
    env_mod.save = False
    trainer_mod.store_value_maps = False
    trainer_mod.train()
    mod_cum_deviation.append(trainer_mod.cumulative_deviation)
    mod_stat_list.append(trainer_mod.stats)

with open(f'mod_deviation_list.pkl', 'wb') as file:
    pickle.dump(mod_cum_deviation, file)
with open(f'mod_stat_list.pkl', 'wb') as file:
    pickle.dump(mod_stat_list, file)


"""## Animation"""

rc('animation', html='jshtml')
plt.rcParams.update({'font.size': 32})
cols = ['b','orange','g','r','purple']
GRIDCOL = 'RdYlGn'

def animate(env_grid, loc_hist, stat_hist, value_hist, skip, file_specifier=None, mod=True, save=True):
    frames = len(loc_hist)
    n_stats = 4
    ax_g = []
    ax_map = []
    filename = 'mod' if mod else 'mon'

    flattened_list = np.array(value_hist).flatten()
    v_min = np.min(flattened_list)
    print(v_min)
    v_max = np.max(flattened_list)
    print(v_max)

    print("Rendering %d frames..." % frames)

    # Set the figure frame
    fig = plt.figure(figsize=(20, 16));
    grids = plt.GridSpec(3, n_stats, hspace=0.2, wspace=0.2);
    for stat in range(n_stats):
        ax_g.append(fig.add_subplot(grids[0,stat]));
        ax_g[stat].set_title(f'Resource {stat+1}',color=cols[stat]);
        ax_g[stat].set_xticks([]);
        ax_g[stat].set_yticks([]);
        if mod:
            ax_map.append(fig.add_subplot(grids[1,stat]));
            ax_map[stat].set_title(f'Value map {stat+1}',color=cols[stat])

    if not mod:
        ax_map = fig.add_subplot(grids[1,:]);
        ax_map.set_title('Value map')

    ax_h = fig.add_subplot(grids[2,:]);

    health_plot = np.zeros((frames, n_stats));

    i=0
    grid_ims = [[] for i in range(n_stats)]
    ims_mod_val = [[] for i in range(n_stats)]
    health = stat_hist[i]
    grid_copy = np.copy(env_grid)

    for stat in range(n_stats):
        grid_ims[stat] = ax_g[stat].imshow(grid_copy[stat]);
        if mod:
            value_map = value_hist[i]
            ims_mod_val[stat] = ax_map[stat].imshow(value_map[stat],cmap=GRIDCOL,vmin=v_min,vmax=v_max)
            ax_map[stat].set_xticks([]);
            ax_map[stat].set_yticks([]);

    if not mod:
        value_map = value_hist[i]
        im_mon_val = ax_map.imshow(value_map,cmap=GRIDCOL,vmin=v_min,vmax=v_max)
        ax_map.set_xticks([]);
        ax_map.set_yticks([]);

    def render_frame(i):
        if i % 1000 == 0:
            print(f"Rendering: time step {i}")

        health = stat_hist[i]
        loc = loc_hist[i]

        # Reset grid_copy to the original env_grid
        grid_copy = np.copy(env_grid)

        # Mark the agent's location
        for stat in range(n_stats):
            grid_copy[stat, loc[0], loc[1]] = -0.02

        # Render grid
        for stat in range(n_stats):
            grid_ims[stat].set_data(grid_copy[stat])
            if mod:
                value_map = value_hist[i]
                display_grid_with_text(value_map[stat], ax_map[stat], ims_mod_val[stat])

        # Render health chart
        health_plot[i] = health
        ax_h.clear()
        ax_h.plot(health_plot[:i], linewidth=5)
        ax_h.plot(range(frames), 5*np.ones(frames), '--g', linewidth=3, alpha=0.5)
        ax_h.axis([0, frames, -5, 10])
        ax_h.set_xlabel('Timestep')
        ax_h.set_ylabel('Stat levels')
        ax_h.set_xticks([])

        if not mod:
            value_map = value_hist[i]
            display_grid_with_text(value_map, ax_map, im_mon_val)

    anim = matplotlib.animation.FuncAnimation(
        fig, render_frame, frames=frames, interval=20
    )
    plt.close()

    # To save the animation
    if save:
        anim.save('movie_clip.mp4', writer='ffmpeg')
    else:
        display(HTML(anim.to_html5_video()))

def display_grid_with_text(grid,ax,im):
      im.set_data(grid)

import copy
import warnings
warnings.filterwarnings( "ignore")
skip = 1
specifier = None
animate(env_mod.original_grid, env_mod.location_history[0:3000], env_mod.stats_history[0:3000],
        trainer_mod.value_map_history[0:3000], skip, specifier, mod=True, save=True)

import warnings
warnings.filterwarnings( "ignore")
skip = 1
specifier = None
animate(env_mono.original_grid, env_mono.location_history, env_mono.stats_history,
        trainer.value_map_history, skip, specifier, mod=False)
