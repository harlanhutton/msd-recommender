from pyspark.ml.feature import StringIndexer
from pyspark import SparkContext
from pyspark.sql import SparkSession
import sys
import getpass


def main(spark, sc, train_input, test_input, val_input,user_id):
    
          
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
    indexer_model_2 = indexer_obj_2.fit(indexer_df_1)
    indexer_df_2 = indexer_model_2.transform(indexer_df_1)
    print('finised indexer 2 on train')

    train_df = indexer_df_2.drop('user_id')
    train_df = train_df.drop('track_id')
    train_df = train_df.repartition(2000)
    
    print('dropped columns in training set')

    val_df_1 = indexer_model_1.transform(valSample)
    ('transform validation set with indexer 1')
    
    val_df_2= indexer_model_2.transform(val_df_1)
    ('transform validation set with indexer 2')

    val_df = val_df_2.drop('user_id')
    val_df= val_df.drop('track_id')
    
    val_df = val_df.repartition(5000)
    #val_df = val_df.checkpoint()

    test_df_1 = indexer_model_1.transform(testSample)
    test_df_2 = indexer_model_2.transform(test_df_1)

    test_df = test_df_2.drop('user_id')
    test_df = test_df.drop('track_id')

    test_df = test_df.repartition(5000)

    test_df.write.mode('overwrite').parquet(f'hdfs:/user/{netID}/pub/test_df.parquet')
    train_df.write.mode('overwrite').parquet(f'hdfs:/user/{netID}/pub/train_df.parquet')
    val_df.write.mode('overwrite').parquet(f'hdfs:/user/{netID}/pub/val_df.parquet')




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