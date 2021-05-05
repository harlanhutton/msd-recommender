import pyspark
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
sparkContext=spark.sparkContext
rdd = sparkContext.parallelize([1,2,3,4,5,6,7,8,9,10])
metrics = RankingMetrics(rdd)
rdd.collect()