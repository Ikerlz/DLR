#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：OptimizationFinal
@File    ：simulation.py
@Author  ：Iker Zhe
@Date    ：2021/2/1 21:12
'''
from util import *
from pyspark import SparkContext
from pyspark import SparkConf
import os
from sklearn.linear_model._coordinate_descent import Lasso

os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk8u252'
os.environ["PYSPARK_PYTHON"] = "/home/lizhe/anaconda3/envs/pyspark/bin/python3.7"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/lizhe/anaconda3/envs/pyspark/bin/python3.7"

if __name__ == '__main__':
    flag = 0
    # shrinkage parameter
    if flag:
        beta = [3, 0, 2, 1.5, 0, 4, 1, 0]
        x_mat, y = simulation_linear(1234, 100000, beta, True)
        n_slices = 5
        dat = np.concatenate((x_mat, y[:, np.newaxis]), axis=1)
        conf = SparkConf().setAppName("LinearRegression_Test").setMaster("local[*]")
        sc = SparkContext.getOrCreate(conf)
        dat_rdd = sc.parallelize(dat, numSlices=n_slices)
        alpha_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 1, 2, 4, 10, 100, 1000, 5000, 10000]
        hat_estimator_mat = np.zeros((len(alpha_list), 8))
        pgd_estimator_mat = np.zeros((len(alpha_list), 8))
        admm_estimator_mat = np.zeros((len(alpha_list), 8))
        sgd_estimator_mat = np.zeros((len(alpha_list), 8))
        for i in range(len(alpha_list)):
            alpha = alpha_list[i]
            # sklearn中的Lasso
            clf = Lasso(alpha=alpha, fit_intercept=False)
            clf.fit(x_mat, y)
            hat_estimator = clf.coef_
            # PGD
            pgd = DlsaLasso(index_x_mat=np.array([0, 1, 2, 3, 4, 5, 6, 7]), index_y=8)
            pgd.method = "PGD"
            pgd.alpha = alpha
            pgd.max_iter = 10000
            pgd_estimator = pgd.dlsa(RDD=dat_rdd)
            # ADMM
            admm = DlsaLasso(index_x_mat=np.array([0, 1, 2, 3, 4, 5, 6, 7]), index_y=8)
            admm.method = "ADMM"
            admm.alpha = alpha
            admm.max_iter = 10000
            admm_estimator = admm.dlsa(RDD=dat_rdd)
            # Subgradient
            sgd = DlsaLasso(index_x_mat=np.array([0, 1, 2, 3, 4, 5, 6, 7]), index_y=8)
            sgd.method = "SubGD"
            sgd.alpha = alpha
            sgd.max_iter = 10000
            sgd_estimator = sgd.dlsa(RDD=dat_rdd)
            hat_estimator_mat[i] = hat_estimator
            pgd_estimator_mat[i] = pgd_estimator
            admm_estimator_mat[i] = admm_estimator
            sgd_estimator_mat[i] = sgd_estimator
            # print("alpha={}".format(alpha))
            # print(hat_estimator)
            # print(pgd_estimator)
            # print(admm_estimator)
            # print(sgd_estimator)
        print(hat_estimator_mat)
        print(pgd_estimator_mat)
        print(admm_estimator_mat)
        print(sgd_estimator_mat)
        np.savetxt("hat_estimator_mat.csv", hat_estimator_mat, delimiter=",")
        np.savetxt("pgd_estimator_mat.csv", pgd_estimator_mat, delimiter=",")
        np.savetxt("admm_estimator_mat.csv", admm_estimator_mat, delimiter=",")
        np.savetxt("sgd_estimator_mat.csv", sgd_estimator_mat, delimiter=",")

    # sample size
    flag = 1
    if flag:
        beta = [3, 2, 1.5, 4, 1]
        n_slices = 5
        conf = SparkConf().setAppName("LinearRegression_Test").setMaster("local[*]")
        sc = SparkContext.getOrCreate(conf)
        total_number_list = [500*x for x in range(1, 21)]
        rmse_mat = np.zeros((20, 5))
        for i in range(20):
            N = total_number_list[i]
            mse_mat = np.zeros((100, 5))
            for j in range(100):
                x_mat, y = simulation_linear(None, N, beta, True)
                dat = np.concatenate((x_mat, y[:, np.newaxis]), axis=1)
                dat_rdd = sc.parallelize(dat, numSlices=n_slices)
                pgd = DlsaLasso(index_x_mat=np.array([0, 1, 2, 3, 4]), index_y=5)
                pgd.max_iter = 10000
                pgd_estimator = pgd.dlsa(RDD=dat_rdd)
                mse_mat[j] = pgd_estimator - beta
            print("进度{}%".format(5*(i+1)))
            print(mse_mat)
            name = "scenario2N_{}.csv".format(N)
            np.savetxt(name, mse_mat, delimiter=",")
            rmse_mat[i] = np.sum(mse_mat**2, axis=0) / 100
        print(rmse_mat)

    # worker num
    flag = 0
    if flag:
        beta = [3, 2, 1.5, 4, 1]
        conf = SparkConf().setAppName("LinearRegression_Test").setMaster("local[*]")
        sc = SparkContext.getOrCreate(conf)
        worker_num_list = [5*x for x in range(1, 21)]
        rmse_mat = np.zeros((20, 5))
        for i in range(20):
            n_slices = worker_num_list[i]
            mse_mat = np.zeros((100, 5))
            for j in range(100):
                x_mat, y = simulation_linear(None, 1000, beta, True)
                dat = np.concatenate((x_mat, y[:, np.newaxis]), axis=1)
                dat_rdd = sc.parallelize(dat, numSlices=n_slices)
                pgd = DlsaLasso(index_x_mat=np.array([0, 1, 2, 3, 4]), index_y=5)
                pgd.max_iter = 10000
                pgd_estimator = pgd.dlsa(RDD=dat_rdd)
                mse_mat[j] = pgd_estimator - beta
            print("进度{}%".format(5*(i+1)))
            print(mse_mat)
            name = "scenario3K_{}.csv".format(n_slices)
            np.savetxt(name, mse_mat, delimiter=",")
            rmse_mat[i] = np.sum(mse_mat**2, axis=0) / 100
        print(rmse_mat)
