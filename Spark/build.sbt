name := "firstApplication"

version := "0.1"

scalaVersion := "2.12.14"

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % "3.1.2",
	"org.apache.spark" %% "spark-sql" % "3.1.2",
	"org.apache.spark" %% "spark-mllib" % "3.1.2",
	"org.postgresql" % "postgresql" % "42.2.21"
)
