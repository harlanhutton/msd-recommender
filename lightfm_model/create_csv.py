from pyspark.sql import SparkSession

def main(spark, filepath, filename):

	df = spark.read.parquet(filepath)
	df.coalesce(1).write.option("header", "true").csv(filename)


if __name__ == "__main__":
    
    # Create the spark session object
    spark = SparkSession.builder.config('spark.driver.memory', '16g')\
    .appName('create csv').getOrCreate()
    
    # Create spark context
    sc = spark.sparkContext

    filepath = sys.argv[1]
    filename = sys.argv[2]
    
    # Call main function
    main(spark, filepath, filename)