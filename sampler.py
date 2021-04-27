from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

import sys

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd

def main(spark, file_path, pct_sample, net_id):

    lines = spark.read.parquet(file_path)
    lines.createOrReplaceTempView('lines')

    df = lines.sample(fraction=pct_sample, seed = 1)
    df.write.mode('overwrite').parquet(f'hdfs:/user/{net_id}/train_sample{pct_sample}.parquet')
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('sampler').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]
    pct_sample = sys.argv[2]
    
    netID = getpass.getuser()

    main(spark, file_path, pct_sample, net_id)
