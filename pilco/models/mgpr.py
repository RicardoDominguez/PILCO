import tensorflow as tf
import numpy as np
import gpflow
float_type = gpflow.settings.dtypes.float_type


class MGPR(gpflow.Parameterized):
    def __init__(self, indices, name=None):
        super(MGPR, self).__init__(name)
        
        self.dyno = indices["dyno"]
        self.angi = indices["angi"]
        self.dyni = indices["dyni"]
        self.poli = indices["poli"]
        self.difi = indices["difi"]
        self.udim = indices["udim"]

        self.num_outputs = self.dyno.shape[0]
        self.num_dims = self.dyni.shape[0]

    def create_models(self, X, Y):
        self.models = []
        X_, Y_ = self.trigAug(X, Y)
        for i in range(self.num_outputs):
            kern = gpflow.kernels.RBF(input_dim=X.shape[1], ARD=True)
            #TODO: Maybe fix noise for better conditioning
            self.models.append(gpflow.models.GPR(X_, Y_[:, i:i+1], kern))
            self.models[i].clear(); self.models[i].compile()
    
    def trigAug(self, X, Y):
        angles = X[:, self.angi]
        sines = np.sin(angles)
        cosines = np.cos(angles)
        sincos = np.empty(X.shape[0], self.angi.shape[0]*2)
        for i in range(self.angi.shape[0]):
            sincos[:, i*2:i*2+1] = np.hstack((sines[:, i], cosines[:, i]))        
        Xaug = np.hstack((X[:, self.dyno], sincos, X[:, :-self.udim]))
        
        Yd = np.copy(Y)
        Yd[:, self.difi] -= X[:, self.difi]
        
        return Xaug[:, self.dyni], Yd[:, self.dyno]
                
    def set_XY(self, X, Y):
        X_, Y_ = self.trigAug(X, Y)
        for i in range(len(self.models)):
            self.models[i].X = X_
            self.models[i].Y = Y_[:, i:i+1]

    def optimize(self):
        optimizer = gpflow.train.ScipyOptimizer(options={'maxfun': 500})
        for model in self.models:
            optimizer.minimize(model)

    def predict_on_noisy_inputs(self, m, s):
        iK, beta = self.calculate_factorizations()
        return self.predict_given_factorizations(m, s, iK, beta)
    
    def calculate_factorizations(self):
        K = self.K(self.X)
        batched_eye = tf.eye(tf.shape(self.X)[0], batch_shape=[self.num_outputs], dtype=float_type)
        L = tf.cholesky(K + self.noise[:, None, None]*batched_eye)
        iK = tf.cholesky_solve(L, batched_eye)
        Y_ = tf.transpose(self.Y)[:, :, None]
        # Why do we transpose Y? Maybe we need to change the definition of self.Y() or beta?
        beta = tf.cholesky_solve(L, Y_)[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        s = tf.tile(s[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
        inp = tf.tile(self.centralized_input(m)[None, :, :], [self.num_outputs, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = tf.matrix_diag(1/self.lengthscales)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + tf.eye(self.num_dims, dtype=float_type)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = tf.linalg.transpose(
                tf.matrix_solve(B, tf.linalg.transpose(iN), adjoint=True),
            )

        lb = tf.exp(-tf.reduce_sum(iN * t, -1)/2) * beta
        tiL = t @ iL
        c = self.variance / tf.sqrt(tf.linalg.det(B))

        M = (tf.reduce_sum(lb, -1) * c)[:, None]
        V = tf.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R = s @ tf.matrix_diag(
                1/tf.square(self.lengthscales[None, :, :]) +
                1/tf.square(self.lengthscales[:, None, :])
            ) + tf.eye(self.num_dims, dtype=float_type)

        # TODO: change this block according to the PR of tensorflow. Maybe move it into a function?
        X = inp[None, :, :, :]/tf.square(self.lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :]/tf.square(self.lengthscales[None, :, None, :])
        Q = tf.matrix_solve(R, s)/2
        Xs = tf.reduce_sum(X @ Q * X, -1)
        X2s = tf.reduce_sum(X2 @ Q * X2, -1)
        maha = -2 * tf.matmul(X @ Q, X2, adjoint_b=True) + \
            Xs[:, :, :, None] + X2s[:, :, None, :]
        #
        k = tf.log(self.variance)[:, None] - \
            tf.reduce_sum(tf.square(iN), -1)/2
        L = tf.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (tf.tile(beta[:, None, None, :], [1, self.num_outputs, 1, 1])
                @ L @
                tf.tile(beta[None, :, :, None], [self.num_outputs, 1, 1, 1])
            )[:, :, 0, 0]

        diagL = tf.transpose(tf.linalg.diag_part(tf.transpose(L)))
        S = S - tf.diag(tf.reduce_sum(tf.multiply(iK, diagL), [1, 2]))
        S = S / tf.sqrt(tf.linalg.det(R))
        S = S + tf.diag(self.variance)
        S = S - M @ tf.transpose(M)

        return tf.transpose(M), S, tf.transpose(V)

    def centralized_input(self, m):
        return self.X - m

    def K(self, X1, X2=None):
        return tf.stack(
            [model.kern.K(X1, X2) for model in self.models]
        )

    @property
    def Y(self):
        return tf.concat(
            [model.Y.parameter_tensor for model in self.models],
            axis = 1
        )

    @property
    def X(self):
        return self.models[0].X.parameter_tensor

    @property
    def lengthscales(self):
        return tf.stack(
            [model.kern.lengthscales.constrained_tensor for model in self.models]
        )

    @property
    def variance(self):
        return tf.stack(
            [model.kern.variance.constrained_tensor for model in self.models]
        )

    @property
    def noise(self):
        return tf.stack(
            [model.likelihood.variance.constrained_tensor for model in self.models]
        )
