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


def main(spark, sc, train_input, test_input, val_input,user_id):
    
    # set up checkpoints
    #sparkContext = spark.sparkContext
    #sc.setCheckpointDir(f'hdfs:/user/{user_id}/checkpoints/')
    
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
    
    train_df = train_df.repartition(50000)
    #train_df = train_df.checkpoint()
    
    print('dropped columns in training set')

    val_df_1 = indexer_model_1.transform(valSample)
    ('transform validation set with indexer 1')
    
    val_df_2= indexer_model_2.transform(val_df_1)
    ('transform validation set with indexer 2')

    val_df = val_df_2.drop('user_id')
    val_df= val_df.drop('track_id')
    
    val_df = val_df.repartition(50000)
    #val_df = val_df.checkpoint()

    test_df_1 = indexer_model_1.transform(testSample)
    test_df_2 = indexer_model_2.transform(test_df_1)

    test_df = test_df_2.drop('user_id')
    test_df = test_df.drop('track_id')

    test_df = test_df.repartition(50000)

    #test_df = test_df.checkpoint()
    
    print('dropped columns in validation set')

        # Import the requisite items

    als = ALS(userCol="user_id_numer",itemCol="track_id_numer",ratingCol="count",
                         coldStartStrategy="drop",implicitPrefs=True,rank=int(20),regParam=float(0.1))


    # Add hyperparameters and their respective values to param_grid
    # param_grid = ParamGridBuilder() \
    #             .addGrid(als.rank, [10, 50, 100, 150]) \
    #             .addGrid(als.regParam, [.01, .05, .1, .15]) \
    #             .build()
    #             #             .addGrid(als.maxIter, [5, 50, 100, 200]) \

               
    # Define evaluator as RMSE and print length of evaluator
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction") 
    # #print ("Num models to be tested: ", len(param_grid))

    # # Build cross validation using CrossValidator
    # cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

    #     #Fit cross validator to the 'train' dataset
    # model = cv.fit(train_df)

    #Extract best model from the cv model above
    # best_model = model.bestModel

    #     # Print best_model
    # print(type(best_model))

    # # Complete the code below to extract the ALS model parameters
    # print("**Best Model**")

    # # # Print "Rank"
    # print("  Rank:", best_model._java_obj.parent().getRank())

    # # Print "MaxIter"
    # print("  MaxIter:", best_model._java_obj.parent().getMaxIter())

    # # Print "RegParam"
    # print("  RegParam:", best_model._java_obj.parent().getRegParam())

    #Hyperparam Tuning
    #tuning_params = {"rank":(30,70),"maxIter":(8,16),"regParam":(.01,1),"alpha":(0.0,3.0)}

    #def BO_func(rank,maxIter,regParam,alpha):
        #recommender = ALS(userCol="user_id_numer",itemCol="track_id_numer",ratingCol="count",
                         #coldStartStrategy="drop",implicitPrefs=True,rank=int(rank),
                         #maxIter=int(maxIter),regParam=int(regParam),alpha=int(alpha))
        #model = recommender.fit(train_df)
        #preds = model.transform(val_df)
        #printres_valid = RegressionEvaluator(metricName="rmse",labelCol="count",
                                       #predictionCol="prediction")
        #rmse=res_valid.evaluate(preds)
        #return rmse

    #optimizer  = BayesianOptimization(f=BO_func, pbounds=tuning_params, verbose=5, random_state=5)
    #optimizer.maximize(init_points=2, n_iter=5)
    #print(optimizer.max)

    #params = optimizer.max.get('params')
    #alpha = params.get("alpha")
    #rank = params.get("rank")
    #maxIter = params.get("maxIter")
    #regParam= params.get("regParam")  

    #implement with optimal hyperparameters
    #recommender = ALS(userCol="user_id_numer",itemCol="track_id_numer",ratingCol="count",
                         #coldStartStrategy="drop",implicitPrefs=True,rank=int(rank),
                         #maxIter=float(maxIter),regParam=float(regParam),alpha=float(alpha))

    #model = best_model.fit(train_df)
    #change the val_df to test
    print("model trained")
    best_model = als.fit(train_df)
    print("fitted")
    test_transformed = best_model.transform(test_df)

#     # Build the recommendation model using ALS on the training data
#     # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

    

#     # use model to transform validation dataset
    #val_transformed = val_transformed.checkpoint()
    
    print('validation set transformed')

#     # for each user, sort track ids by count
    test_true = test_df.orderBy('count')
    #test_true.show()

#     # flatten to group by user id and get list of true track ids
    test_true_flatten = test_true.groupby('user_id_numer').agg(func.collect_list('track_id_numer').alias("track_id_numer"))
    print('validation set flattened')
#     # add to dictionary

    test_true_flatten.cache()
    test_true_dict = test_true_flatten.collect()
    
    #val_true_dict.show()
    test_true_dict = [{r['user_id_numer']: r['track_id_numer']} for r in test_true_dict]
    test_true_dict = dict((key,d[key]) for d in test_true_dict for key in d)
    
    print('created val true dictionary')

#     # get distinct users from transformed test set
    users = test_transformed.select(als.getUserCol()).distinct()

#     # get predictions for test users
    test_preds = model.recommendForUserSubset(users,500)
    test_preds_explode = test_preds.select(test_preds.user_id_numer,func.explode(test_preds.recommendations.track_id_numer))
    test_preds_flatten = test_preds_explode.groupby('user_id_numer').agg(func.collect_list('col').alias("col"))

#     # add test predictions to dictionary
    test_preds_flatten.cache()
    test_preds_dict = test_preds_flatten.collect()
    test_preds_dict = [{r['user_id_numer']: r['col']} for r in test_preds_dict]
    test_preds_dict = dict((key,d[key]) for d in test_preds_dict for key in d)

    print('created val preds dictionary')

   
#     ### NEW WAY to create predictions and labels 
    dictcon= list(map(list, test_preds_dict.items()))
    dfpreds = spark.createDataFrame(dictcon, ["user_id_numer", "tracks"])

    dfpreds = dfpreds.repartition(50000)

    dictcon2= list(map(list, test_true_dict.items()))
    dftrue = spark.createDataFrame(dictcon2, ["user_id_numer", "tracks"])

    print('created val true and val preds df')
    
    dftrue = dftrue.repartition(50000)


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
