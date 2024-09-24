import gym
from gym import spaces
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt

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
        self.save = True
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