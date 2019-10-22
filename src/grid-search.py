from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.recommendation import ALS
import pandas as pd
spark = SparkSession.builder.getOrCreate()

estimator = ALS()
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
paramGrid = ParamGridBuilder().addGrid(estimator.rank, [2,5,8,10,15,20,25])\
                              .addGrid(estimator.userCol, ['userId'])\
                              .addGrid(estimator.itemCol, ['movieId'])\
                              .addGrid(estimator.ratingCol, ['rating'])\
                              .addGrid(estimator.maxIter, [10,15,20])\
                              .addGrid(estimator.regParam, [.05,.1,.15,.20,.25,.3])\
                              .addGrid(estimator.nonnegative, [True])\
                              .addGrid(estimator.coldStartStrategy, ['drop'])\
                              .build()
pipeline = Pipeline(stages=[estimator])

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

train_pandas_df = pd.read_csv('data/movies/ratings_train.csv',index_col=0)
train_df = spark.createDataFrame(train_pandas_df)

cvModel = crossval.fit(train_df)

params = [{p.name: v for p, v in m.items()} for m in cvModel.getEstimatorParamMaps()]

results = pd.DataFrame.from_dict([
    {cvModel.getEvaluator().getMetricName(): metric, **ps} 
    for ps, metric in zip(params, cvModel.avgMetrics)])

results.to_csv('data/movies/als-grid-search.csv')