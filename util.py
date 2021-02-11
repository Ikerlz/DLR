import numpy as np
import warnings
from numpy.linalg import norm, cholesky


def simulation_linear(random_seed, total_number, param_vec, bias=False):
    """
    :param random_seed: the random state;
    :param total_number: the sample size;
    :param param_vec: the vector of model coefficients;
    :param bias: there is a bias or not;
    :return: a dataset which satisfies the params above.
    """
    np.random.seed(random_seed)
    p = len(param_vec)
    if bias:
        x_mat0 = np.ones((total_number, 1))
        x_mat = np.hstack((x_mat0, np.random.randn(total_number, p - 1)))
    else:
        x_mat = np.random.randn(total_number, p)
    response = x_mat.dot(param_vec) + np.random.randn(total_number)
    return x_mat, response


class LassoSolver(object):
    """Lasso regression.

        Minimizes the objective function::

                ||y - Xw||^2_2 + alpha * ||w||_1

        Parameters
        ----------
        alpha : float, default=1.0
            Constant that multiplies the penalty terms. Defaults to 1.0.
            See the notes for the exact mathematical meaning of this
            parameter. ``alpha = 0`` is equivalent to an ordinary least square

        fit_intercept : bool, default=True
            Whether the intercept should be estimated or not. If ``False``, the
            data is assumed to be already centered.

        max_iter : int, default=1000
            The maximum number of iterations

        tol : float, default=1e-4
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.

        method : str, default=PGD
            The method to solve the lasso problem including "PGD", "ADMM", "SubGD".

        Attributes
        ----------
        coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
            parameter vector (w in the cost function formula)

        sparse_coef_ : sparse matrix of shape (n_features, 1) or \
                (n_targets, n_features)
            ``sparse_coef_`` is a readonly property derived from ``coef_``

        intercept_ : float or ndarray of shape (n_targets,)
            independent term in decision function.

        n_iter_ : list of int
            number of iterations run by the coordinate descent solver to reach
            the specified tolerance.


        Notes
        -----
        To avoid unnecessary memory duplication the X argument of the fit method
        should be directly passed as a Fortran-contiguous numpy array.
        """

    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000,
                 step_size=0.01, tol=1e-4, method="PGD"):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.step_size = step_size
        self.tol = tol
        self.method = method

    def fit(self, x_mat, y):
        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the Linear Regression "
                          "estimator", stacklevel=2)
        n = x_mat.shape[0]
        dim = x_mat.shape[1]
        inverse_covariance_mat = x_mat.T @ x_mat
        alpha = self.step_size  # 固定步长
        p = self.alpha  # 正则化参数
        epsilon = self.tol  # 最大允许误差
        x_k = np.zeros(dim)
        x_k_old = np.zeros(dim)

        k = 1  # 迭代次数
        if self.method == "PGD":
            print("======= Method: PGD =======")
            while k < self.max_iter:
                x_k_temp = x_k - alpha * x_mat.T @ (x_mat @ x_k - y)  # 对光滑部分做梯度下降

                # 临近点投影（软阈值）
                for i in range(dim):
                    if x_k_temp[i] < -alpha * p:
                        x_k[i] = x_k_temp[i] + alpha * p
                    elif x_k_temp[i] > alpha * p:
                        x_k[i] = x_k_temp[i] - alpha * p
                    else:
                        x_k[i] = 0

                if np.linalg.norm(x_k - x_k_old) < epsilon:
                    break
                else:
                    x_k_old = x_k.copy()  # 深拷贝
                    k += 1
            x_optm = x_k[:]
            print(k)
        elif self.method == "SubGD":
            print("======= Method: SubGD =======")
            while k < self.max_iter:
                # 计算目标函数次梯度
                l1_subgradient = np.zeros(dim)  # L1范数的次梯度
                for i in range(len(x_k)):
                    if x_k[i] != 0:
                        l1_subgradient[i] = np.sign(x_k[i])
                    else:
                        l1_subgradient[i] = np.random.uniform(-1, 1)  # 随机取[-1, 1]内的值作为次梯度
                subgradient = x_mat.T @ (x_mat @ x_k - y) + p * l1_subgradient

                x_k = x_k - alpha * subgradient
                if np.linalg.norm(x_k - x_k_old) < epsilon:
                    break
                else:
                    x_k_old = x_k.copy()  # 深拷贝
                    k += 1
            x_optm = x_k.copy()  # 最优解
            print(k)
        elif self.method == "ADMM":
            print("\n======= Method: ADMM =======")

            # define functions
            def factor(X, rho):
                m, n = X.shape
                if m >= n:
                    R = cholesky(X.T.dot(X) + rho * np.eye(n))
                else:
                    R = cholesky(np.eye(m) + 1. / rho * (X.dot(X.T)))
                return R.T

            def prox(vec, mu):
                y = np.maximum(np.abs(vec) - mu, np.zeros((dim, 1)))
                return np.sign(vec) * y

            z = np.zeros((dim, 1))
            x_old = np.zeros((dim, 1))
            u = np.zeros((dim, 1))
            rho = 0.01
            sm = rho
            rel_par = 1.618
            Atb = x_mat.T.dot(y).reshape((dim, 1))
            # AtA = x_mat.T.dot(x_mat)
            R = factor(x_mat, rho)

            while k < self.max_iter:
                # update x
                w = Atb + sm * z - u
                x = np.linalg.inv(R.T.dot(R)).dot(w)

                # update z
                c = x + u / sm
                z = prox(c, p / sm)

                # update u
                u = u + rel_par * sm * (x - z)

                if np.linalg.norm(x - x_old) < epsilon:
                    break
                else:
                    x_old = x.copy()  # 深拷贝
                    k += 1
            x_optm = z
            print(k)

            # rho = 1
            # rel_par = 1
            #
            # # define functions
            # def factor(X, rho):
            #     m, n = X.shape
            #     if m >= n:
            #         L = cholesky(X.T.dot(X) + rho * sparse.eye(n))
            #     else:
            #         L = cholesky(sparse.eye(m) + 1. / rho * (X.dot(X.T)))
            #     L = sparse.csc_matrix(L)
            #     U = sparse.csc_matrix(L.T)
            #     return L, U
            #
            # def shrinkage(x, kappa):
            #     return np.maximum(0., x - kappa) - np.maximum(0., -x - kappa)
            #
            # # save a matrix-vector multiply
            # Xty = x_mat.T.dot(y)
            #
            # # ADMM solver
            # z_old = np.zeros((dim, 1))
            # z = np.zeros((dim, 1))
            # u = np.zeros((dim, 1))
            #
            # # cache the (Cholesky) factorization
            # L, U = factor(x_mat, rho)
            # while k < self.max_iter:
            #     # x-update
            #     q = Xty + rho * (z - u)  # (temporary value)
            #     if n >= dim:
            #         x = spsolve(U, spsolve(L, q))[..., np.newaxis][0]
            #     else:
            #         ULXq = spsolve(U, spsolve(L, x_mat.dot(q)))[..., np.newaxis]
            #         x = (q * 1. / rho) - ((x_mat.T.dot(ULXq)) * 1. / (rho ** 2))
            #     # z-update with relaxation
            #     zold = np.copy(z)
            #     x_hat = rel_par * x + (1. - rel_par) * zold
            #     z = shrinkage(x_hat + u, alpha * 1. / rho)
            #
            #     # u-update
            #     u += (x_hat - z)
            #
            #     if np.linalg.norm(z - zold) < epsilon:
            #         break
            #     k += 1
            #
            # x_optm = z.ravel()

        else:
            raise ValueError(
                "The value of method should be one of the PGD, ADMM, SubGD, however {} is given".format(self.method))
        return n, x_optm, inverse_covariance_mat


class DlsaLasso(LassoSolver):
    """
    :Perform linear regression in a distributed manner.
    :Note: The DLSA algorithm used here can be found in the reference below.
    :Reference: Xuening Zhu, Feng Li, Hansheng Wang
    :"Least Squares Approximation for a Distributed System", arXiv:1908.04904v2
    """

    def __init__(self, index_x_mat, index_y):
        """
        :param index_x_mat: the column index of independent variables used in the model;
        :param index_y: the column index of dependent variable used in the model;
        :param fit_intercept: If True, the intercept will be considered.
        :param theta_init: the initial value of theta;
        :param sample_size: the sample size of the data;
        :param rounds: rounds of TDLSA.
        """
        super().__init__()
        self.index_x_mat = index_x_mat
        self.index_y = index_y

    # def linear_regression(self, x_mat, y):
    #     """
    #     :param x_mat: the matrix of independent variables;
    #     :param y: the dependent variable;
    #     :return: a list of
    #         * n: the sample size;
    #         * theta_hat: the estimation of coefficients;
    #         * inverse_covariance_mat: the inverse covariance matrix.
    #     """
    #     n = x_mat.shape[0]
    #     dim = x_mat.shape[1]
    #     inverse_covariance_mat = x_mat.T @ x_mat
    #     alpha = 0.001  # 固定步长
    #     p = 0.5  # 正则化参数
    #     epsilon = 1e-5  # 最大允许误差
    #
    #     x_k = np.zeros(dim)
    #     x_k_old = np.zeros(dim)
    #
    #     k = 1  # 迭代次数
    #
    #     while k < 1e6:
    #         x_k_temp = x_k - alpha * x_mat.T @ (x_mat @ x_k - y)  # 对光滑部分做梯度下降
    #
    #         # 临近点投影（软阈值）
    #         for i in range(dim):
    #             if x_k_temp[i] < -alpha * p:
    #                 x_k[i] = x_k_temp[i] + alpha * p
    #             elif x_k_temp[i] > alpha * p:
    #                 x_k[i] = x_k_temp[i] - alpha * p
    #             else:
    #                 x_k[i] = 0
    #
    #         if np.linalg.norm(x_k - x_k_old) < epsilon:
    #             break
    #         else:
    #             x_k_old = x_k.copy()  # 深拷贝
    #             k += 1
    #
    #     return n, x_k[:], inverse_covariance_mat

    def linear_iter_function(self, iterator):
        """
        :param iterator: iterator of partitions in the RDD;
        :return: a list of
            * n_k: the sample size in the partition;
            * theta_hat_k: the estimation of coefficients obtained from the partition;
            * inverse_covariance_mat_k: the inverse covariance matrix obtained from the partition.
        """
        dat_iter = np.array([*iterator])
        x_mat_k = dat_iter[:, self.index_x_mat]
        y_k = dat_iter[:, self.index_y]
        n_k, theta_hat_k, inverse_covariance_mat_k = LassoSolver.fit(self, x_mat=x_mat_k, y=y_k)
        return n_k, theta_hat_k, inverse_covariance_mat_k

    def distributed_linear_regression(self, RDD):
        """
        :param RDD: Resilient Distributed Datasets;
        :return: a 3*k numpy array of n_k, theta_hat_k, inverse_covariance_mat_k in each partition.
        """
        output_worker = RDD.mapPartitions(self.linear_iter_function).collect()
        result_worker = np.array(output_worker).reshape(-1, 3).T
        return result_worker

    def dlsa(self, RDD):
        """
        """
        worker_df = self.distributed_linear_regression(RDD)
        num_in_worker_list = worker_df[0].tolist()
        total_num = sum(num_in_worker_list)
        self.sample_size = total_num
        theta_list = worker_df[1].tolist()
        sigma_inv_list = worker_df[2].tolist()
        worker_num = len(theta_list)
        num_of_variable = len(theta_list[0])
        sigma_inv = np.zeros((num_of_variable, num_of_variable))
        sigma_inv_sum = np.zeros((num_of_variable, num_of_variable))
        sig_theta = np.zeros((num_of_variable, 1))
        for i in range(worker_num):
            alpha = num_in_worker_list[i] / total_num
            sigma_k = sigma_inv_list[i]
            theta_k = theta_list[i].reshape(num_of_variable, 1)
            sigma_inv_sum = sigma_inv_sum + sigma_k
            sigma_inv = sigma_inv + alpha * sigma_k
            sig_theta = sig_theta + alpha * np.dot(sigma_k, theta_k)
        theta_wlse = np.dot(np.linalg.inv(sigma_inv), sig_theta).reshape(-1, )
        self.theta_init = theta_wlse
        return theta_wlse
