import numpy as np

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