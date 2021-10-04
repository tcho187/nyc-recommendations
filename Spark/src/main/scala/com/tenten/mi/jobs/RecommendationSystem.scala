package com.tenten.mi.jobs

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

object RecommendationSystem extends App with RecommendationWrapper {
	import session.implicits._

	val movielens_ratings: StructType = StructType(Array(
		StructField("index", IntegerType, true),
		StructField("movieId", IntegerType, true),
		StructField("rating", DoubleType, true),
		StructField("userId",  IntegerType, true),
	))

	val ratingsDf = buildDataFrame("ratings", movielens_ratings)
	ratingsDf.show(10)

	val Array(training, test) = ratingsDf.randomSplit(Array(0.8, 0.2))


	// Build the recommendation model using ALS on the training data
	val als = new ALS()
		.setMaxIter(5)
		.setRegParam(0.01)
		.setUserCol("userId")
		.setItemCol("movieId")
		.setRatingCol("rating")
	val model = als.fit(training)

	// Evaluate the model by computing the RMSE on the test data
	// Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
	model.setColdStartStrategy("drop")
	val predictions = model.transform(test)
	val evaluator = new RegressionEvaluator()
		.setMetricName("rmse")
		.setLabelCol("rating")
		.setPredictionCol("prediction")
	val rmse = evaluator.evaluate(predictions)
	println(s"Root-mean-square error = $rmse")

	// Generate top 10 movie recommendations for each user
	val userRecs = model.recommendForAllUsers(10)
	// Generate top 10 user recommendations for each movie
	val movieRecs = model.recommendForAllItems(10)
	session.stop()
}
