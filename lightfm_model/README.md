# LightFM Model

To compare to our ALS model, we sampled 1% of each parquet file for train, validation, and test. 
We created CSVs for the 1% parquet files to use in our lightfm model implementation.
For both models, we timed both hyperparameter tuning as well as final model fitting and prediction.
We used the same hyperparameter ranges for both models.
Finally, we compare both execution time and MAP scores.
