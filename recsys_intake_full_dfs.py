from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as func
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext
from pyspark.sql import SparkSession
import sys
import getpass


def main(spark, sc):

    print('set up spark context')

    #sc.setCheckpointDir('hdfs:/user/ahh303/checkpoints')
    
    # Read in parquet files
    train_df = spark.read.parquet('hdfs:/user/ahh303/pub/train_df_full.parquet')
    test_df = spark.read.parquet('hdfs:/user/ahh303/pub/test_df_full.parquet')
    val_df = spark.read.parquet('hdfs:/user/ahh303/pub/val_df_full.parquet')

    train_df.createOrReplaceTempView('train_df')
    test_df.createOrReplaceTempView('test_df')
    val_df.createOrReplaceTempView('val_df')

#     train_df.checkpoint()
#     test_df.checkpoint()
#     val_df.checkpoint()

    print('dfs created')

    # Create ALS Model object
    als = ALS(userCol="user_id_numer",itemCol="track_id_numer",ratingCol="count",
                         coldStartStrategy="drop",implicitPrefs=True,rank=int(30),regParam=float(0.1),maxIter=30)


    print("model created")

    # Fit model
    best_model = als.fit(train_df)

    print("model fitted")
    
    # Select users from test df
    users = test_df.select(als.getUserCol()).distinct()

    print('users selected')

    # Create predictions for test users
    test_preds = best_model.recommendForUserSubset(users,500)
    test_preds_explode = test_preds.select(test_preds.user_id_numer,func.explode(test_preds.recommendations.track_id_numer))
    test_preds_flatten = test_preds_explode.groupby('user_id_numer').agg(func.collect_list('col').alias("col"))

    print('test preds df created')
    
    # Create dataframe for true test user listens
    test_true_flatten = test_df.groupby('user_id_numer').agg(func.collect_list('track_id_numer').alias("track_id_numer"))
#     test_true_flatten.checkpoint()

    print('test true df created')
 
    # Create RDD of predictions and true listens
    rankingsRDD = (test_preds_flatten.join(test_true_flatten, 'user_id_numer').rdd.map(lambda row: (row[1], row[2])))

    print('RDD created')
    
    # Call Ranking Metrics on predictions and true
    metrics = RankingMetrics(rankingsRDD)

    print("Ranking Metrics called")

    # Get Mean Average Precision
    MAP = metrics.meanAveragePrecision
    print(MAP)


if __name__ == "__main__":
    
    # Create the spark session object
    spark = SparkSession.builder.config('spark.executor.instances', '8')\
    .config('spark.executor.memory', '16g')\
    .config('spark.driver.memory', '8g')\
    .config('spark.executor.cores', '3')\
    .config('spark.default.parallelism', '48')\
    .appName('sampler').getOrCreate()
    
    # Create spark context
    sc = spark.sparkContext
    
    # Call main function
    main(spark, sc)
