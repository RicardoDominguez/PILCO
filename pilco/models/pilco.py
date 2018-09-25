import numpy as np
import tensorflow as tf
import gpflow
import pandas as pd

from .mgpr import MGPR
from .smgpr import SMGPR
from .. import controllers
from .. import rewards

float_type = gpflow.settings.dtypes.float_type


class PILCO(gpflow.models.Model):
    def __init__(self, indices, num_induced_points=None, horizon=30, controller=None,
                reward=None, m_init=None, S_init=None, name=None):
        super(PILCO, self).__init__(name)
        
        self.dyno = indices["dyno"]
        self.angi = indices["angi"]
        self.dyni = indices["dyni"]
        self.poli = indices["poli"]
        self.difi = indices["difi"]
        self.udim = indices["udim"]
        
        if not num_induced_points:
            self.mgpr = MGPR(indices)
        else:
            self.mgpr = SMGPR(indices, num_induced_points)
        self.horizon = horizon

        if controller is None:
            self.controller = controllers.LinearController(self.poli.shape[0], 
                                                           self.udim)
        else:
            self.controller = controller

        if reward is None:
            self.reward = rewards.ExponentialReward(self.dyno.shape[0])
        else:
            self.reward = reward
        
        if m_init is None:
            # If the user has not provided an initial state for the rollouts,
            # then use a zero array.
            self.m_init = np.zeros(self.dyno.shape[0])
        else:
            self.m_init = m_init
            
        if S_init is None:
            self.S_init = np.diag(np.ones(self.dyno.shape[0]) * 0.1)
        else:
            self.S_init = S_init

    @gpflow.name_scope('likelihood')
    def _build_likelihood(self):
        # This is for tuning controller's parameters
        reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
        return reward

    def optimize(self):
        '''
        Optimizes both GP's and controller's hypeparamemeters.
        '''
        import time
        start = time.time()
        self.mgpr.optimize()
        end = time.time()
        print("Finished with GPs' optimization in %.1f seconds" % (end - start))
        start = time.time()
        optimizer = gpflow.train.ScipyOptimizer(options={'maxfun': 500})
        optimizer.minimize(self, disp=True)
        end = time.time()
        print("Finished with Controller's optimization in5%.1f seconds" % (end - start))

        lengthscales = {}; variances = {}; noises = {};
        i = 0
        for model in self.mgpr.models:
            lengthscales['GP' + str(i)] = model.kern.lengthscales.value
            variances['GP' + str(i)] = np.array([model.kern.variance.value])
            noises['GP' + str(i)] = np.array([model.likelihood.variance.value])
            i += 1

        print('-----Learned models------')
        pd.set_option('precision', 3)
        print('---Lengthscales---')
        print(pd.DataFrame(data=lengthscales))
        print('---Variances---')
        print(pd.DataFrame(data=variances))
        print('---Noises---')
        print(pd.DataFrame(data=noises))

    @gpflow.autoflow((float_type,[None, None]))
    def compute_action(self, x_m):
        return self.controller.compute_action(
                x_m, tf.zeros([self.poli.shape[0], self.poli.shape[0]], 
                              float_type))[0]

    def predict(self, m_x, s_x, n):
        loop_vars = [
            tf.constant(0, tf.int32),
            m_x,
            s_x,
            tf.constant([[0]], float_type)
        ]

        _, m_x, s_x, reward = tf.while_loop(
            # Termination condition
            lambda j, m_x, s_x, reward: j < n,
            # Body function
            lambda j, m_x, s_x, reward: (
                j + 1,
                *self.propagate(m_x, s_x),
                tf.add(reward, self.reward.compute_reward(m_x, s_x)[0])
            ), loop_vars
        )

        return m_x, s_x, reward

    def propagate(self, m_x, s_x):
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        m = tf.concat([m_x, m_u], axis=1)
        s1 = tf.concat([s_x, s_x@c_xu], axis=1)
        s2 = tf.concat([tf.transpose(s_x@c_xu), s_u], axis=1)
        s = tf.concat([s1, s2], axis=0)

        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_x = M_dx + m_x
        #TODO: cleanup the following line
        S_x = S_dx + s_x + s1@C_dx + tf.matmul(C_dx, s1, transpose_a=True, transpose_b=True)

        # While-loop requires the shapes of the outputs to be fixed
        M_x.set_shape([1, self.dyno.shape[0]])
        S_x.set_shape([self.dyno.shape[0], self.dyno.shape[0]])
        return M_x, S_x
