from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

lines = spark.read.parquet(file_path)
parts = lines.map(lambda row: row.value.split("::"))
print(parts)