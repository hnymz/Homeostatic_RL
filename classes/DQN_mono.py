from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

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
            print(f"Shape of mask: {self.mask.shape}")
            print(f"Shape of state: {state.shape}")
            state = state * self.mask

        # Pass the input state through the first hidden layer with ReLU activation
        x = F.relu(self.fc1(state))
        # Pass the result through the second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Pass the result through the output layer to get Q-values for each action
        return self.fc3(x)

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

class DQNAgent:
    def __init__(self, env, policy, action_size, learning_rate, gamma, lambd, w, blind = False, hidden_size=64):
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
        self.w = w

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
            next_q_values = self.w * self.target_network(next_states).gather(  # Double DQN
                1, self.q_network(next_states).argmax(dim=1, keepdim=True)
            ).squeeze(1).detach() + (1 - self.w) * self.target_network(next_states).gather(
                1, self.q_network(next_states).argmin(dim=1, keepdim=True)
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

            if frame % 1000 == 0:
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
