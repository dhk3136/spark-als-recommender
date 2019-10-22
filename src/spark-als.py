import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import mean,isnan,col
spark = SparkSession.builder.getOrCreate()

class SparkALS():

    def train(self,train_df,regParam,maxIter,rank):
        self.train_df = train_df
        model = ALS(userCol='userId',itemCol='movieId',ratingCol='rating',nonnegative=True,regParam=regParam,maxIter=maxIter,rank=rank)
        self.recommender = model.fit(train_df)

    def predict(self,test_df):
        self.predictions = self.recommender.transform(test_df)

    def evaluate(self):

        mean_rating = train_df.select([mean('rating')]).collect()
        mean_rating
        results={}
        for i in mean_rating:
            results.update(i.asDict())
        the_mean = results['avg(rating)']
        new_predictions = self.predictions.na.fill({'prediction': the_mean})
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
        return evaluator.evaluate(new_predictions)

if __name__ == '__main__':
    train_df = spark.read.format("csv").option("header", "true").option('inferSchema','true').load("data/movies/ratings_train.csv")
    test_df = spark.read.format("csv").option("header", "true").option('inferSchema','true').load("data/movies/ratings_test.csv")

    gridlist = []

    for rank in range(20):
        if rank == 0: rank += 1
        for j in range(20):
            if j == 0: j += 1
            regParam = j/20
            test = SparkALS()
            test.train(train_df,regParam=regParam,maxIter=22,rank=rank)
            test.predict(test_df)
            rmse = test.evaluate()
            gridlist.append([rank,regParam,rmse])
            print(f"Rank: {rank}, regParam: {regParam}, RMSE: {rmse}")
            np.savetxt("3dgraphdata.csv", gridlist, delimiter=",")