from util import *
from pyspark import SparkContext
from pyspark import SparkConf
import os


os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk8u252'
os.environ["PYSPARK_PYTHON"]="/home/lizhe/anaconda3/envs/pyspark/bin/python3.7"
os.environ["PYSPARK_DRIVER_PYTHON"]="/home/lizhe/anaconda3/envs/pyspark/bin/python3.7"

# Spark Environment
# findspark.init("/usr/local/spark")

if __name__ == '__main__':
    # settings
    theta = [3, 0, 0, 1.5, 0, 0, 2, 0]
    x_mat, y = simulation_linear(123, 100, theta, True)
    # test = LassoSolver(method="ADMM")
    # res = test.fit(x_mat, y)
    # print(res)
    n_slices = 4
    dat = np.concatenate((x_mat, y[:, np.newaxis]), axis=1)
    conf = SparkConf().setAppName("LinearRegression_Test").setMaster("local[*]")
    sc = SparkContext.getOrCreate(conf)
    dat_rdd = sc.parallelize(dat, numSlices=n_slices)
    linear_regression_test = DlsaLasso(index_x_mat=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                                       index_y=8)
    linear_regression_test.method = "ADMM"
    print(linear_regression_test.dlsa(RDD=dat_rdd))