from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import getpass
import sys

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd

import sqlite3

from pyspark.sql import SQLContext


def main(spark):
    
    df = 'track_metadata.db'
    
    with sqlite3.connect(df) as conn:
        
        cursor = conn.cursor()
    
        cursor.execute("SELECT * FROM sqlite_master WHERE type='table'")
        
        #cursor.execute('SELECT * FROM tracks')
        
        print(cursor.all())
        
        #print(cursor.fetchone()[0])

#         df_sample = df.sample(fraction=.01, seed = 1)

#         df_sample.write.mode('overwrite').parquet('hdfs:/user/jke261/meta_db_sample.parquet')

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object

    spark = SparkSession.builder.appName('sampler_db').getOrCreate()
    
    main(spark)
