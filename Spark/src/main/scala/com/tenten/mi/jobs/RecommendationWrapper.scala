package com.tenten.mi.jobs

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._

trait RecommendationWrapper {
	lazy val session = SparkSession
		.builder
		.master("local[*]")
		.appName("RS")
		.getOrCreate()

	session.sparkContext.setLogLevel("WARN")

//	val dataSetSchema: StructType



	def buildDataFrame(dataSet: String, dataSetSchema: StructType): DataFrame = {
		session.read
			.format("jdbc")
			.option("url", "")
			.option("driver", "org.postgresql.Driver")
			.option("dbtable", s"public.$dataSet")
			.option("user", "")
			.option("password", "")
			.option("header", true).schema(dataSetSchema).option("nullValue", "")
			.option("treatEmptyValuesAsNulls", "true")
			.load(dataSet).cache()
	}
}
