from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as func
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import sys
import getpass


def main(spark, sc ,user_id):
    
    # set up checkpoints
    #sparkContext = spark.sparkContext
    #sc.setCheckpointDir(f'hdfs:/user/{user_id}/checkpoints/')
    
    print('set up spark context')
    
    
    train_df = spark.read.parquet('train_df.parquet')
    test_df = spark.read.parquet('test_df.parquet')
    val_df = spark.read.parquet('val_df.parquet')
    
    val_df.createOrReplaceTempView('val_df')
    train_df.createOrReplaceTempView('train_df')
    test_df.createOrReplaceTempView('test_df')
   

        # Import the requisite items

    als = ALS(userCol="user_id_numer",itemCol="track_id_numer",ratingCol="count",
                         coldStartStrategy="drop",implicitPrefs=True,rank=int(20),regParam=float(0.1))



    print("model trained")
    best_model = als.fit(train_df)
    print("fitted")
    
    users = test_df.select(als.getUserCol()).distinct()

    test_preds = best_model.recommendForUserSubset(users,500)
    test_preds_explode = test_preds.select(test_preds.user_id_numer,func.explode(test_preds.recommendations.track_id_numer))
    test_preds_flatten = test_preds_explode.groupby('user_id_numer').agg(func.collect_list('col').alias("col"))
    test_true_flatten = test_df.groupby('user_id_numer').agg(func.collect_list('track_id_numer').alias("track_id_numer"))
    #test_true_flatten = test_true_flatten.repartition(5000)
    rankingsRDD = (test_preds_flatten.join(test_true_flatten, 'user_id_numer').rdd.map(lambda row: (row[1], row[2])))
    metrics = RankingMetrics(rankingsRDD)

    print("Ranking Metrics called")

    MAP = metrics.meanAveragePrecision
    print(MAP)


if __name__ == "__main__":
        # Create the spark session object
    spark = SparkSession.builder.appName('sampler').getOrCreate()

    # Get file_path for dataset to analyze
#     train = sys.argv[1]
#     test = sys.argv[2]
#     val = sys.argv[3]
    
    sc = spark.sparkContext
    
    netID = getpass.getuser()
    
    main(spark, sc, netID)
