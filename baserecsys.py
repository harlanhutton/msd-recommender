from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

import sys
import bench

from pyspark.sql import SparkSession
import numpy as np


def main(spark, file_path):

	lines = spark.read.parquet(file_path)
	lines.createOrReplaceTempView('lines')
	print(lines[1:10])


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)