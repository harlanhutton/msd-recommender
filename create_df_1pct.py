from pyspark.ml.feature import StringIndexer
from pyspark import SparkContext
from pyspark.sql import SparkSession
import sys
import getpass


def main(spark):
           
    # read in data
    trainSample = spark.read.parquet('train_sample1.parquet')
    testSample = spark.read.parquet('test_sample1.parquet')
    valSample = spark.read.parquet('val_sample1.parquet')
    
    valSample.createOrReplaceTempView('valSample')
    trainSample.createOrReplaceTempView('trainSample')
    testSample.createOrReplaceTempView('testSample')

#     # StringIndexer to create new columns and dataframes
    indexer_obj_1 = StringIndexer(inputCol="user_id", outputCol="user_id_numer").setHandleInvalid("keep")
    indexer_model_1 = indexer_obj_1.fit(trainSample)
    indexer_df_1 = indexer_model_1.transform(trainSample)
    print('finished indexer 1 on train')

    indexer_obj_2 = StringIndexer(inputCol="track_id", outputCol="track_id_numer").setHandleInvalid("keep")
    indexer_model_2 = indexer_obj_2.fit(indexer_df_1)
    indexer_df_2 = indexer_model_2.transform(indexer_df_1)
    print('finished indexer 2 on train')

    train_df = indexer_df_2.drop('user_id')
    train_df = train_df.drop('track_id')
    train_df = train_df.repartition(5000)
    
    print('dropped columns in training set')

    val_df_1 = indexer_model_1.transform(valSample)
    print('transform validation set with indexer 1')
    
    val_df_2= indexer_model_2.transform(val_df_1)
    print('transform validation set with indexer 2')

    val_df = val_df_2.drop('user_id')
    val_df= val_df.drop('track_id')
    
    val_df = val_df.repartition(5000)
    #val_df = val_df.checkpoint()

    test_df_1 = indexer_model_1.transform(testSample)
    test_df_2 = indexer_model_2.transform(test_df_1)

    test_df = test_df_2.drop('user_id')
    test_df = test_df.drop('track_id')

    test_df = test_df.repartition(5000)

    test_df.write.mode('overwrite').parquet('test_df1.parquet')
    train_df.write.mode('overwrite').parquet('train_df1.parquet')
    val_df.write.mode('overwrite').parquet('val_df1.parquet')




if __name__ == "__main__":
        # Create the spark session object
    spark = SparkSession.builder.appName('sampler').getOrCreate()
    
    main(spark)