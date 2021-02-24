#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：OptimizationFinal 
@File    ：realdata.py
@Author  ：Iker Zhe
@Date    ：2021/2/16 23:36 
'''
from util import *
from pyspark import SparkContext
from pyspark import SparkConf
import os
import pandas as pd
from sklearn.linear_model._coordinate_descent import Lasso
from sklearn import preprocessing
import numpy as np

os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk8u252'
os.environ["PYSPARK_PYTHON"] = "/home/lizhe/anaconda3/envs/pyspark/bin/python3.7"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/lizhe/anaconda3/envs/pyspark/bin/python3.7"


if __name__ == '__main__':
    data = pd.read_csv("usercar.csv")
    y = data["price"].values
    y = (y - np.mean(y)) / np.sqrt(np.cov(y))
    data = data.drop(["name", "price"], axis=1)
    x_mat = data.values
    for i in range(28):
        x_mat[:, i] = (x_mat[:, i] - np.mean(x_mat[:, i])) / np.sqrt(np.cov(x_mat[:, i]))
    param_vec = [0 for _ in range(28)]
    n_slices = 3
    alpha = 0.001
    index_vec = [x for x in range(28)]
    dat = np.concatenate((x_mat, y[:, np.newaxis]), axis=1)
    conf = SparkConf().setAppName("RealData").setMaster("local[*]")
    sc = SparkContext.getOrCreate(conf)
    dat_rdd = sc.parallelize(dat, numSlices=n_slices)
    # sklearn中的Lasso
    clf = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    clf.fit(x_mat, y)
    hat_estimator = clf.coef_
    print(hat_estimator)
    # PGD
    print("PGD")
    pgd = DlsaLasso(index_x_mat=np.array(index_vec), index_y=28)
    pgd.method = "PGD"
    pgd.alpha = alpha
    pgd.max_iter = 10000
    pgd_estimator = pgd.dlsa(RDD=dat_rdd)
    print(pgd_estimator)
    # ADMM
    print("ADMM")
    admm = DlsaLasso(index_x_mat=np.array(index_vec), index_y=28)
    admm.method = "ADMM"
    admm.alpha = alpha
    admm.max_iter = 10000
    admm_estimator = admm.dlsa(RDD=dat_rdd)
    print(admm_estimator)
    # Subgradient
    print("Subgradient")
    sgd = DlsaLasso(index_x_mat=np.array(index_vec), index_y=28)
    sgd.method = "SubGD"
    sgd.alpha = alpha
    sgd.max_iter = 10000
    sgd_estimator = sgd.dlsa(RDD=dat_rdd)
    print(sgd_estimator)

