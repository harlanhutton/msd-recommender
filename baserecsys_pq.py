from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

import sys

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd

import os

def main(spark, file_path):

	#os.chdir('hdfs://user/ahh303/final-project-recommender-systers/')

	#print(os.getcwd())

	lines = spark.read.parquet(file_path)
	lines.createOrReplaceTempView('lines')
    print(type(lines), 'lines type')
    print(lines.shape,'lines shape)
	df = lines.sample(fraction=0.01, seed = 1)
    print(type(df), 'dftype')
    print(df.shape, 'shape')
	df.write.mode('overwrite').parquet(f'hdfs:/user/jke261/test01sample.parquet')


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
