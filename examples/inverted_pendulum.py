import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
np.random.seed(0)

env = gym.make('InvertedPendulum-v2')

def rollout(policy, timesteps):
    latent = []
    env.reset()
    x, _, _, _ = env.step(0)
    for timestep in range(timesteps):
        #env.render()
        u = policy(x)
        x_new, _, done, _ = env.step(u)
        latent.append(np.hstack((x, u)))
        if done: 
            latent.append(np.hstack((x_new, u)))
            break
        x = x_new
    print("HHHH:",  timestep)
    return np.stack(latent[:-1, :]), np.stack(latent[1:, :])

def random_policy(x):
    return env.action_space.sample()

def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]


# Full state representation
# 1 x           cart position
# 2 v           cart velocity
# 3 theta       angle of the pendulum
# 4 dtheta      angular velocity
# 5 sin(theta)  complex representation...
# 6 cos(theta)  of theta
# 7 u           force applied to cart
    
# Important indices
indices = {
    "dyno": np.array([1, 2, 3, 4]), # variables to be predicted (and known to loss)
    "angi": np.array([3]), # angle variables
    "dyni": np.array([1, 2, 4, 5, 6]), # variables that are inputs to the dynamics GP
    "poli": np.array([1, 2, 4, 5, 6]), # variables that are inputs to the policy
    "difi": np.array([1, 2, 3, 4]), # variables that are learned via differences
    "udim": 1, # dimension of control action
}

# Controller
controller = RbfController(indices, num_basis_functions=5)

# Build PILCO instance
#m_init = # 
pilco = PILCO(indices, controller=controller, horizon=40)

# Initial random rollouts to generate a dataset
X, Y = rollout(policy=random_policy, timesteps=40)
print(X[0, :])
for i in range(1,3):
    X_, Y_ = rollout(policy=random_policy, timesteps=40)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))


# Example of user provided reward function, setting a custom target state
# R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
# pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

# Example of fixing a parameter, optional, for a linear controller only
#pilco.controller.b = np.array([[0.0]])
#pilco.controller.b.trainable = False

pilco.mgpr.create_models(X, Y)
for rollouts in range(3):
    pilco.optimize()
    import pdb; pdb.set_trace()
    X_new, Y_new = rollout(policy=pilco_policy, timesteps=100)
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)
