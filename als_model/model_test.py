from pyspark.ml.recommendation import ALSModel
from pyspark.sql import Row
import pyspark.sql.functions as func
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext
from pyspark.sql import SparkSession

def main(spark, sc):

	# load test df
	test_df = spark.read.parquet('hdfs:/user/ahh303/pub/test_df_full.parquet')
	test_df.createOrReplaceTempView('test_df')

	# load trained model
	model = ALSModel.load('hdfs:/user/ahh303/pub/model')

	# Select users from test df
    users = test_df.select('user_id_numer').distinct()

    print('users selected')

    # Create predictions for test users
    test_preds = model.recommendForUserSubset(users,500)
    test_preds = test_preds.select(test_preds.user_id_numer,func.explode(test_preds.recommendations.track_id_numer))
    test_preds = test_preds.groupby('user_id_numer').agg(func.collect_list('col').alias("col"))

    print('test preds df created')
    
    # Create dataframe for true test user listens
    test_true = test_df.groupby('user_id_numer').agg(func.collect_list('track_id_numer').alias("track_id_numer"))

    print('test true df created')
 
    # Create RDD of predictions and true listens
    recs_and_true_RDD = (test_preds.join(test_true, 'user_id_numer').rdd.map(lambda row: (row[1], row[2])))

    print('RDD created')
    
    # Call Ranking Metrics on predictions and true
    metrics = RankingMetrics(recs_and_true_RDD)

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
    .config('spark.default.parallelism', '48')
    .appName('sampler').getOrCreate()
    
    # Create spark context
    sc = spark.sparkContext
    
    # Call main function
    main(spark, sc)
