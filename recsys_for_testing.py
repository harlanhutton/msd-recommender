from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as func
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import sys
import getpass


def main(spark, sc, train_input, test_input, val_input,user_id):
    
    # set up checkpoints
    #sparkContext = spark.sparkContext
    sc.setCheckpointDir(f'hdfs:/user/{user_id}/final_project/checkpoints')
    
    print('set up spark context')
          
          
    # read in data
    trainSample = spark.read.parquet(train_input)
    testSample = spark.read.parquet(test_input)
    valSample = spark.read.parquet(val_input)
    
    valSample.createOrReplaceTempView('valSample')
    trainSample.createOrReplaceTempView('trainSample')
    testSample.createOrReplaceTempView('testSample')

#     # StringIndexer to create new columns and dataframes
    indexer_obj_1 = StringIndexer(inputCol="user_id", outputCol="user_id_numer").setHandleInvalid("keep")
    indexer_model_1 = indexer_obj_1.fit(trainSample)
    indexer_df_1 = indexer_model_1.transform(trainSample)
    print('finised indexer 1 on train')

    indexer_obj_2 = StringIndexer(inputCol="track_id", outputCol="track_id_numer").setHandleInvalid("keep")
    indexer_model_2= indexer_obj_2.fit(indexer_df_1)
    indexer_df_2 = indexer_model_2.transform(indexer_df_1)
    print('finised indexer 2 on train')

    train_df = indexer_df_2.drop('user_id')
    train_df= train_df.drop('track_id')
    
    train_df = train_df.repartition(500)
    
    print('dropped columns in training set')

    val_df_1 = indexer_model_1.transform(valSample)
    ('transform validation set with indexer 1')
    
    val_df_2= indexer_model_2.transform(val_df_1)
    ('transform validation set with indexer 2')

    val_df = val_df_2.drop('user_id')
    val_df= val_df.drop('track_id')
    
    val_df = val_df.repartition(500)
    
    print('dropped columns in validation set')

#     # Build the recommendation model using ALS on the training data
#     # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id_numer", itemCol="track_id_numer", ratingCol= "count",
              coldStartStrategy="drop", implicitPrefs = True)
    model = als.fit(train_df)

    print("model trained")

#     # use model to transform validation dataset
    val_transformed = model.transform(val_df)
    
    print('validation set transformed')

#     # for each user, sort track ids by count
    val_true = val_df.orderBy('count')
    val_true.show()

#     # flatten to group by user id and get list of true track ids
    val_true_flatten = val_true.groupby('user_id_numer').agg(func.collect_list('track_id_numer').alias("track_id_numer"))
    print('validation set flattened')
#     # add to dictionary
    val_true_dict = val_true_flatten.collect()
    
    #val_true_dict.show()
    val_true_dict = [{r['user_id_numer']: r['track_id_numer']} for r in val_true_dict]
    val_true_dict = dict((key,d[key]) for d in val_true_dict for key in d)
    
    print('created val true dictionary')

#     # get distinct users from transformed validation set
    users = val_transformed.select(als.getUserCol()).distinct()

#     # get predictions for validation users
    val_preds = model.recommendForUserSubset(users, 10)
    val_preds_explode = val_preds.select(val_preds.user_id_numer,func.explode(val_preds.recommendations.track_id_numer))
    val_preds_flatten = val_preds_explode.groupby('user_id_numer').agg(func.collect_list('col').alias("col"))

#     # add validation predictions to dictionary
    val_preds_dict = val_preds_flatten.collect()
    val_preds_dict = [{r['user_id_numer']: r['col']} for r in val_preds_dict]
    val_preds_dict = dict((key,d[key]) for d in val_preds_dict for key in d)

    print('created val preds dictionary')

   
#     ### NEW WAY to create predictions and labels 
    dictcon= list(map(list, val_preds_dict.items()))
    dfpreds = spark.createDataFrame(dictcon, ["user_id_numer", "tracks"])

    dictcon2= list(map(list, val_true_dict.items()))
    dftrue = spark.createDataFrame(dictcon2, ["user_id_numer", "tracks"])

    print('created val true and val preds df')
    
    #dftrue = dftrue.repartition(500)


    rankingsRDD = (dfpreds.join(dftrue, 'user_id_numer').rdd.map(lambda row: (row[1], row[2])))

    print('created RDD')

    metrics = RankingMetrics(rankingsRDD)

    print("Ranking Metrics called")
    
    MAP = metrics.meanAveragePrecision
    print(MAP)

if __name__ == "__main__":
        # Create the spark session object
    spark = SparkSession.builder.appName('sampler').getOrCreate()

    # Get file_path for dataset to analyze
    train = sys.argv[1]
    test = sys.argv[2]
    val = sys.argv[3]
    
    sc = spark.sparkContext
    
    netID = getpass.getuser()
    
    main(spark, sc, train, test, val, netID)
