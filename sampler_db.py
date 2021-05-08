from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import getpass
import sys

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd


def main(spark, netID):
    
    filer.createOrReplaceTempView('/scratch/work/public/MillionSongDataset/')
    
    sampler = spark.sql('SELECT * FROM filer ORDER BY RANDOM() LIMIT 2000')
    
    sampler.write.mode('overwrite').parquet(f'hdfs:/user/{netID}/meta_db_sample.parquet')


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('sampler_db').getOrCreate()
    
    netID = getpass.getuser()

    main(spark, netID)
