import sys
import matplotlib.pyplot as plt
import arviz as az
import pickle
import torch

from stats import AgentStats
from environment import HomeostaticEnv
from DQN_mono import ReplayBuffer, EpsilonGreedyPolicy, DQNAgent, DQNTrainer

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

"""## Run"""

# Set parameters for the agent and environment
n_stats = 4
n_frames = 30000

stat_manager = AgentStats(
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
    pq=[4, 2],
    mental_increase=0.05,
    initial_mental=5,
    mental_health_range=[3, 5]
)

env_mono = HomeostaticEnv(
    stat_manager,
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
    resource_percentile=95,
    pain_percentile=70,
    pain_loc=[2, 8],
    n_pain_stats=1
)

replay_buffer = ReplayBuffer(buffer_size=100000, batch_size=512)

policy = EpsilonGreedyPolicy(
    decay_rate=decay,
    decision_process="None",
    epsilon_start=1,
    epsilon_end=0.01
)

mono_agent = DQNAgent(
    env_mono,
    policy,
    action_size=env_mono.n_actions,
    learning_rate=0.001,
    gamma=0.5,
    lambd=0.0001,
    w=w,
    blind=True,
    hidden_size=512
)

trainer_mono = DQNTrainer(
    env_mono,
    mono_agent,
    replay_buffer,
    target_update_freq=200,
    plotting=True,
    num_frames=n_frames,
    ep_length=100000
)

# Start training
env_mono.save = True
trainer_mono.store_value_maps = False
trainer_mono.train()

with open(f'../RL/outputs/mon/mono_stat_{decay}_w{w}_{batch}.pkl', 'wb') as file:
    pickle.dump(trainer_mono.stats, file)
with open(f'../RL/outputs/mon/mono_location_{decay}_w{w}_{batch}.pkl', 'wb') as file:
    pickle.dump(env_mono.location_history, file)
with open(f'../RL/outputs/mon/mono_cumu_deviation_{decay}_w{w}_{batch}.pkl', 'wb') as file:
    pickle.dump(trainer_mono.cumulative_deviation, file)

# Plotting the lines
az.style.use("arviz-doc")
plt.figure(figsize=(10, 6))
plt.plot(trainer_mono.stats)
plt.axhline(y=5, color='grey', linestyle='--')
plt.xlabel('Time step')
plt.ylabel('Stat value')
plt.title('Monolithic')
plt.legend([f'Stat {i}' for i in range(n_stats)])
plt.savefig(f'../RL/figures/mon/mono_stat_{decay}_w{w}_{batch}.png')
plt.close()

plt.imshow(env_mono.heat_map)
plt.savefig(f'../RL/figures/mon/mono_heatmap_{decay}_w{w}_{batch}.png')
plt.close()