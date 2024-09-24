import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt

"""# PPO

## Buffer
"""

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

"""## Network"""

def layer_init(model, std=np.sqrt(2), bias_const=0.0):
    for layer in model.children():
        try:
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
        except:
            pass
    return model

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        print(f"statedim: {state_dim}")
        print(f"hiddendim: {hidden_dim}")

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, action_dim),
                        nn.Softmax(dim=-1)
                    )

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, 1)
                    )

        layer_init(self.actor)
        layer_init(self.critic)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def mod_prob(self, state):
        action_probs = self.actor(state)
        state_val = self.critic(state)
        return action_probs.detach(), state_val.detach()

    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

"""## PPO agent"""

class PPO:
    def __init__(self, env, entropy_coeff_start, entropy_coeff_end, entropy_decay,
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip, hidden_dim):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coeff = entropy_coeff_start
        self.entropy_coeff_end = entropy_coeff_end
        self.entropy_decay = entropy_decay

        self.state_dim = self.env.n_statedims
        self.action_dim = self.env.n_actions
        self.hidden_dim = hidden_dim
        self.is_modular = False

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()


    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()


    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        if rewards.numel() !=1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        with torch.no_grad():
            old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
            old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards.detach() - old_state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - self.entropy_coeff*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        self.entropy_coeff = max(self.entropy_coeff_end, self.entropy_coeff - self.entropy_decay)

        return loss.mean().detach().numpy()


"""## Trainer"""

class PPOTrainer:
    """
    - Train agents.
    - Visualize
    """
    def __init__(self, env, agent, update_freq=200, plotting=True, num_frames=30000, ep_length=100000):
        self.env = env
        self.agent = agent
        self.num_frames = num_frames
        self.update_freq = update_freq
        self.ep_length = ep_length

        # Visualization
        self.plotting_interval = 100
        self.plotting = plotting

    def train(self):
        self.stats = []
        self.rewards = []
        self.mod_rewards = []
        self.losses = []
        self.entropy = []

        for frame in range(1, self.num_frames + 1):
            state = self.env.reset().unsqueeze(0)  # Reset environment to get initial state
            done = False

            if frame % 1000 == 0:
                print(f"Training: time step {frame}")

            # Get action, state and reward
            action = self.agent.select_action(state)  # Select action based on current policy
            next_state, reward, done, sep_rewards = self.env.step(action, self.ep_length)

            self.rewards.append(reward)
            self.mod_rewards.append(sep_rewards)
            self.stats.append(copy.deepcopy(self.env.stats))

            # Store reward
            if self.agent.is_modular == False:
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)
            else:
                for i in range(self.agent.numModules):
                    self.agent.buffers[i].rewards.append(sep_rewards[i])
                    self.agent.buffers[i].is_terminals.append(done)

            state = next_state.unsqueeze(0)  # Update state

            # Update networks
            if frame % self.update_freq == 0:
                loss = self.agent.update()
                self.losses.append(loss)
                self.entropy.append(self.agent.entropy_coeff)

            # Visualization
            if frame % self.plotting_interval == 0 and self.plotting:
                self.plot(frame)
                self.env.render()
                plt.show()

    def plot(
            self,
            frame
            ):
        """
        Plot the training progresses.
        """
        plt.figure(figsize=(24, 5))
        if self.agent.is_modular == False:
            plt.subplot(131)
            plt.title('rewards')
            plt.plot(self.rewards)
        else:
            plt.subplot(131)
            plt.title('module rewards')
            plt.plot(self.mod_rewards)

        plt.subplot(132)
        plt.title('losses')
        plt.plot(self.losses)

        plt.subplot(133)
        plt.title('stats')
        plt.plot(self.stats)

        plt.ylim([-10, 10])
        plt.legend([f'stat {i + 1}: {np.round(self.env.stats[i], 2)}' for i in range(self.env.n_stats)])
        plt.xlabel('Time step')
        plt.ylabel('Stat levels')
        plt.show()