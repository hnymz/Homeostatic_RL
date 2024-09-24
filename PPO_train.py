import sys
import matplotlib.pyplot as plt
import arviz as az
import pickle
import torch

from stats import AgentStats
from environment import HomeostaticEnv
from PPO import PPO, PPOTrainer

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

"""## Run"""

# Configuration parameters
n_frames = 70000
n_stats = 4
lr = float(sys.argv[1])
Kepoch = int(sys.argv[2])
gamma = float(sys.argv[3])
hidden = int(sys.argv[4])
eps_clip = float(sys.argv[5])
update_freq = int(sys.argv[6])

print(f"Processing: lr-{lr} kepoch-{Kepoch} gamma-{gamma} hidden-{hidden} eps_clip-{eps_clip} update_freq-{update_freq}")

# Initialize AgentStats
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
    pq=[4, 2]
)

# Initialize HomeostaticEnv
env_ppo = HomeostaticEnv(
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
    resource_percentile=95
)

# Initialize PPO agent
ppo_agent = PPO(
    env_ppo,
    entropy_coeff_start=0.1,
    entropy_coeff_end=0.1,
    entropy_decay=0.001,
    lr_actor=lr,
    lr_critic=lr,
    gamma=gamma,
    K_epochs=Kepoch,
    eps_clip=eps_clip,
    hidden_dim=hidden
)

# Initialize PPO Trainer
trainer_ppo = PPOTrainer(
    env_ppo,
    ppo_agent,
    update_freq=update_freq,
    plotting=False,
    num_frames=n_frames,
    ep_length=100000
)

# Train the PPO agent
trainer_ppo.train()

# Save results
with open(f'/RL/outputs/ppo/ppo_location_{lr}_{gamma}_{Kepoch}_{hidden}_{eps_clip}_{update_freq}.pkl', 'wb') as file:
    pickle.dump(env_ppo.location_history, file)

with open(f'/RL/outputs/ppo/ppo_stats_{lr}_{gamma}_{Kepoch}_{hidden}_{eps_clip}_{update_freq}.pkl', 'wb') as file:
    pickle.dump(trainer_ppo.stats, file)

"""## Plot"""

# Plotting the lines
az.style.use("arviz-doc")
plt.figure(figsize=(10, 6))
plt.plot(trainer_ppo.stats)
plt.axhline(y=5, color='grey', linestyle='--')
plt.xlabel('Time step')
plt.ylabel('Stat value')
plt.title('PPO')
plt.legend([f'Stat {i}' for i in range(n_stats)])
plt.savefig(f'/RL/figures/ppo/stat_{lr}_{gamma}_{Kepoch}_{hidden}_{eps_clip}_{update_freq}.png')
plt.close()

# Save heatmap
plt.imshow(env_ppo.heat_map)
plt.savefig(f'/RL/figures/ppo/heatmap_{lr}_{gamma}_{Kepoch}_{hidden}_{eps_clip}_{update_freq}.png')
plt.close()
