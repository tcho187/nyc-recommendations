package com.tenten.mi.jobs

import org.apache.spark.ml.Pipeline
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, NGram, Tokenizer}
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.types.{BooleanType, DoubleType, IntegerType, StringType, StructField, StructType, TimestampType}
import org.apache.spark.sql.functions.{col, explode, filter, lit, lower, regexp_replace, trim, udf}
import org.sparkproject.dmg.pmml.True

object TikTok extends App with RecommendationWrapper {

	import session.implicits._

	val stickersSchema: StructType = StructType(Array(
		StructField("id", IntegerType, true),
		StructField("nickname", StringType, true),
		StructField("video_id", StringType, true),
		StructField("sticker_id", IntegerType, true),
		StructField("sticker", StringType, true),
		StructField("create_time", TimestampType, true)
	))


	val restaurantSchema: StructType = StructType(Array(
		StructField("restaurant", StringType, true),
		StructField("cuisine_types", StringType, true),
		StructField("description", StringType, true),
		StructField("price_ranges", StringType, true),
		StructField("aggregate_ratings", StringType, true),
		StructField("source", StringType, true),
		StructField("added_timestamp", TimestampType, true),
		StructField("use_analysis", BooleanType, true)
	))

	var stickersDF = buildDataFrame("stickers", stickersSchema)


	var restaurantDF = buildDataFrame("restaurants", restaurantSchema)
	restaurantDF = restaurantDF.filter($"use_analysis" === true)


	stickersDF = stickersDF.withColumn("newSticker", regexp_replace($"sticker", lit("\uD83D\uDCCD"), lit("")))
	stickersDF = stickersDF.withColumn("newSticker", regexp_replace($"newSticker", lit("NYC"), lit("")))
	stickersDF = stickersDF.withColumn("newSticker", trim(col("newSticker")))
	val tokenizer = new Tokenizer().setInputCol("newSticker").setOutputCol("words")

	val tokenized = tokenizer.transform(stickersDF)
	val ngrams = (1 to 4).map(i =>
		new NGram().setN(i)
			.setInputCol("words").setOutputCol(s"${i}_grams")
	)

	def buildNgrams(inputCol: String = "tokens", outputCol: String = "features", n: Int = 3) = {

		val ngrams = (1 to 4).map(i =>
			new NGram().setN(i)
				.setInputCol("words").setOutputCol(s"${i}_grams")
		)

		new Pipeline().setStages((ngrams).toArray)

	}

	var test = buildNgrams().fit(tokenized).transform(tokenized)
	test = test.withColumn("test", lit(null))
	//	test = test.withColumn("test", explode($"1_grams"))

	test.show(1, truncate = false)

	var apple = session.createDataFrame(session.sparkContext.emptyRDD[Row], test.schema)
	val suffix: String = "_grams"
	var i_grams_Cols: List[String] = Nil
	for (i <- 1 to 3) {
		val iGCS = i.toString.concat(suffix)
		i_grams_Cols = i_grams_Cols ::: List(iGCS)
	}

	for (i <- i_grams_Cols) {
		val nameCol = col({i})
		apple = apple.union(test.withColumn("test", explode({nameCol})))
	}

	apple.show(1,false)

	println(apple.count())

	var joinDF = apple.join(restaurantDF, lower(apple("test")) === lower(restaurantDF("restaurant")))
	joinDF = joinDF.withColumn("eventStrength", lit(1.0))
	joinDF = joinDF.groupBy("nickname", "restaurant", "sticker", "eventStrength").count()

	//	udf to hashcode
	// You could have also defined the UDF this way
	val hashCodeUserID = udf { nickname: String => nickname.hashCode }
	joinDF = joinDF
		.withColumn("userId", hashCodeUserID(joinDF("nickname")))
		.withColumn("restaurantId", hashCodeUserID(joinDF("restaurant")))
	joinDF = joinDF.select("userId","restaurantId","eventStrength")
	joinDF.show(20, false)
	println(joinDF.count())


	val Array(training, testing) = joinDF.randomSplit(Array(0.8, 0.2))

	val als = new ALS()
		.setMaxIter(5)
		.setRegParam(0.01)
		.setUserCol("userId")
		.setItemCol("restaurantId")
		.setRatingCol("eventStrength")

	val model = als.fit(training)


	// Evaluate the model by computing the RMSE on the test data
	// Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
	model.setColdStartStrategy("drop")
	val predictions = model.transform(testing)
	val evaluator = new RegressionEvaluator()
		.setMetricName("rmse")
		.setLabelCol("eventStrength")
		.setPredictionCol("prediction")
	val rmse = evaluator.evaluate(predictions)
	println(s"Root-mean-square error = $rmse")

	// Generate top 10 movie recommendations for each user
	val userRecs = model.recommendForAllUsers(10)
	println(userRecs)
	// Generate top 10 user recommendations for each movie
	val movieRecs = model.recommendForAllItems(10)

}



