from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as func
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np

# create Spark Session 
spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
#sc = spark._sc

# read in data
trainSample = spark.read.parquet('train_sample1.parquet')
testSample = spark.read.parquet('test_sample1.parquet')
valSample = spark.read.parquet('val_sample1.parquet')
valSample.createOrReplaceTempView('valSample')
trainSample.createOrReplaceTempView('trainSample')
testSample.createOrReplaceTempView('testSample')

# StringIndexer to create new columns and dataframes
indexer_obj_1 = StringIndexer(inputCol="user_id", outputCol="user_id_numer").setHandleInvalid("keep")
indexer_model_1 = indexer_obj_1.fit(trainSample)
indexer_df_1 = indexer_model_1.transform(trainSample)

indexer_obj_2 = StringIndexer(inputCol="track_id", outputCol="track_id_numer").setHandleInvalid("keep")
indexer_model_2= indexer_obj_2.fit(indexer_df_1)
indexer_df_2 = indexer_model_2.transform(indexer_df_1)

train_df = indexer_df_2.drop('user_id')
train_df= train_df.drop('track_id')

val_df_1 = indexer_model_1.transform(valSample)
val_df_2= indexer_model_2.transform(val_df_1)

val_df = val_df_2.drop('user_id')
val_df= val_df.drop('track_id')

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="user_id_numer", itemCol="track_id_numer", ratingCol= "count",
          coldStartStrategy="drop", implicitPrefs = True)
model = als.fit(train_df)

# use model to transform validation dataset
val_transformed = model.transform(val_df)

# for each user, sort track ids by count
val_true = val_df.orderBy('count')

# flatten to group by user id and get list of true track ids
val_true_flatten = val_true.groupby('user_id_numer').agg(func.collect_list('track_id_numer').alias("track_id_numer"))

# add to dictionary
val_true_dict = val_true_flatten.collect()
val_true_dict = [{r['user_id_numer']: r['track_id_numer']} for r in val_true_dict]
val_true_dict = dict((key,d[key]) for d in val_true_dict for key in d)


# get distinct users from transformed validation set
users = val_transformed.select(als.getUserCol()).distinct()

# get predictions for validation users
val_preds = model.recommendForUserSubset(users, 10)
val_preds_explode = val_preds.select(val_preds.user_id_numer,func.explode(val_preds.recommendations.track_id_numer))
val_preds_flatten = val_preds_explode.groupby('user_id_numer').agg(func.collect_list('col').alias("col"))

# add validation predictions to dictionary
val_preds_dict = val_preds_flatten.collect()
val_preds_dict = [{r['user_id_numer']: r['col']} for r in val_preds_dict]
val_preds_dict = dict((key,d[key]) for d in val_preds_dict for key in d)

# create predictions and labels RDD and get MAP
labels_list = []

for user in val_preds_dict.keys():
    labels_list.append((val_preds_dict[user][0], [int(i) for i in val_true_dict[user]][0]))

labels = spark.sparkContext.parallelize(labels_list)
metrics = RankingMetrics(labels)
print(metrics.meanAveragePrecision)

