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
    print(type(lines))
    print((lines.count(), len(lines.columns)))
    print('lines type and shape')
    df = lines.sample(fraction=0.01, seed = 1)
    print(type(df), 'dftype')
    print(df.count(), len(df.columns), 'shape')
    df.write.mode('overwrite').parquet(f'hdfs:/user/jke261/test01sample.parquet')
    
    
    re_read = spark.read.parquet('hdfs:/user/jke261/test01sample.parquet')
    re_read.createOrReplaceTempView('re_read')
    print(re_read.head(20))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
