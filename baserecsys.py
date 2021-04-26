from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

import sys

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd


def main(spark, file_path):

	lines = spark.read.parquet(file_path)
	lines.createOrReplaceTempView('lines')
	df = lines.sample(fraction=0.01, seed = 1)
	df.toPandas().to_csv('hdfs:/user/ahh303/train.csv')


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)