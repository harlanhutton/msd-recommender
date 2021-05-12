from pyspark.ml.recommendation import ALS
from pyspark import SparkContext
from pyspark.sql import SparkSession
import sys

def main(spark, sc, rank, reg, m_iter):

    print('set up spark context')

    sc.setCheckpointDir('hdfs:/user/ahh303/checkpoints')

    
    # Read in parquet files
    train_df = spark.read.parquet('hdfs:/user/ahh303/pub/train_df.parquet')
    val_df = spark.read.parquet('hdfs:/user/ahh303/pub/val_df_full.parquet')

    train_df.createOrReplaceTempView('train_df')
    val_df.createOrReplaceTempView('val_df')

    print('DFs created')

    # Create ALS Model object
    als = ALS(userCol="user_id_numer",itemCol="track_id_numer",ratingCol="count",
                         coldStartStrategy="drop",implicitPrefs=True,rank=int(rank),regParam=float(reg),maxIter=int(m_iter))


    print("model created")

    # Fit model
    model = als.fit(train_df)

    print("model fitted")

    model.write().overwrite().save('hdfs:/user/ahh303/pub/model')

if __name__ == "__main__":
    
    # Create the spark session object
    spark = SparkSession.builder.config('spark.executor.instances', '4')\
    .config('spark.executor.memory', '16g')\
    .config('spark.driver.memory', '4g')\
    .config('spark.executor.cores', '4')\
    .config('spark.default.parallelism', '48')\
    .appName('sampler').getOrCreate()
    
    # Create spark context
    sc = spark.sparkContext

    rank = sys.argv[1]
    reg = sys.argv[2]
    m_iter = sys.argv[3]
    
    # Call main function
    main(spark, sc, rank, reg, m_iter)