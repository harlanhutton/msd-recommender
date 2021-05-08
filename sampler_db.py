from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import getpass
import sys

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd

from pyspark.sql import SQLContext


def main(spark):
    
    df = spark.read.format('jdbc').options(driver='org.sqlite.JDBC', dbtable='my_table', url='jdbc:/scratch/work/public/MillionSongDataset/AdditionalFiles/track_metadata.db').load()
    
#      df = sqlContext.read.format('jdbc').\
#      options(url='jdbc:sqlite:/scratch/work/public/MillionSongDataset/AdditionalFiles/track_metadata.db',\
#      dbtable='employee',driver='org.sqlite.JDBC').load()
    
#     sampler = spark.sql("SELECT * FROM `/scratch/work/public/MillionSongDataset/AdditionalFiles/track_metadata.db `")
    
#     sampler.createOrReplaceTempView("sampler")
    
    df_sample = df.sample(fraction=.01, seed = 1)
    
    df_sample.write.mode('overwrite').parquet('hdfs:/user/jke261/meta_db_sample.parquet')

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object


    spark = SparkSession.builder\
               .config('spark.jars.packages', 'org.xerial:sqlite-jdbc:3.34.0')\
               .getOrCreate()


    main(spark)
