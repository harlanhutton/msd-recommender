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


def main(spark, sc):
    
    train_df = spark.read.parquet('train_samplehyp10.parquet')
    test_df = spark.read.parquet('test_samplehyp10.parquet')
    val_df = spark.read.parquet('val_samplehyp10.parquet')

    train_df.createOrReplaceTempView('train_df')
    test_df.createOrReplaceTempView('test_df')
    val_df.createOrReplaceTempView('val_df')

    print('dfs created')
    # Add hyperparameters and their respective values to param_grid
    als = ALS(userCol="user_id_numer",itemCol="track_id_numer",ratingCol="count",
                         coldStartStrategy="drop",implicitPrefs=True,rank=int(20),regParam=float(0.1))
    param_grid = ParamGridBuilder().addGrid(als.rank, [5,10,15,20]).addGrid(als.regParam,[.01,.1,1]).build()
    
    
    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
    
    # # Build cross validation using CrossValidator
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    
    #     #Fit cross validator to the 'train' dataset
    model = cv.fit(train_df)
    
    #Extract best model from the cv model above
    best_model = model.bestModel
    
    #     # Print best_model
    print(type(best_model))
    
    # # Complete the code below to extract the ALS model parameters
    print("**Best Model**")
    
    # # # Print "Rank"
    print("  Rank:", best_model._java_obj.parent().getRank())
    
    # # Print "RegParam"
    print("  RegParam:", best_model._java_obj.parent().getRegParam())

if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('sampler').getOrCreate()
    
    sc = spark.sparkContext
    
    #netID = getpass.getuser()
    
    main(spark, sc)

