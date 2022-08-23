# Databricks notebook source
# MAGIC %md
# MAGIC # Accufly Project Phase 1

# COMMAND ----------

import datetime
import re

from pyspark.sql.functions import col
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import split

from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml import Pipeline

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


import airporttime

# COMMAND ----------

blob_container = "w261accufly" # The name of your container created in https://portal.azure.com
storage_account = "w261accufly" # The name of your Storage acount created in https://portal.azure.com
secret_scope = "w261accufly" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261accufly" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
# blob_url = 'https://w261accufly.blob.core.windows.net/w261accufly?sp=r&st=2022-07-08T20:26:17Z&se=2022-09-02T04:26:17Z&sv=2021-06-08&sr=c&sig=RcZ5srt6hPWivxaGigfs20ttXuMj5naIk0m2AGlMqN8%3D'
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.account.key.w261accufly.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Phase 1 - Summary
# MAGIC In Phase 1, the team compiled a report detailing the project abstract, preliminary analysis of data, initial EDA, machine learning algorithms and standard metrics to be used, pipeline steps, and task splitting for future phases of the project. During this stage of the project, we gained a deeper understanding of the data by taking note of the different column values we will be analyzing, along with identifying any null values and other values necessary to be imputed. We used this information to begin thinking about which columns of the tables we would want to be joining for Phase II. For the machine learning algorithms, we decided to use random forest, logistic regression, and gradient boosted tree ensembles for reasons listed in the Machine Learning Algorithms section, and our metrics used will be accuracy, precision, recall, and F-1 score. We have also included a block diagram that lists the projected schedule of tasks for the rest of the project.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Abstract
# MAGIC 
# MAGIC The flight industry plays a crucial goal in the world’s transportation sector, and lots of businesses and people rely on various airlines to connect to other parts of the world. Unfortunately, flight delays have gradually increased and become inevitable, due to weather conditions or operational issues, which incur significant financial losses for many airline companies. To resolve this issue, our team will build a model to accurately predict flight delays with our main customer of the project being the consumer. We define a delay as a 15-minute delay (or greater) from the original time of departure, and the prediction should be performed at least 2 hours before departure time. We will analyze past trends of weather conditions and affected flights to build a model which will allow passengers to be prepared for any changes to their plans and enables airlines to proactively respond to potential flight delays to diminish the negative impact. Our evaluation metrics are F1 score using precision and recall. For Phase 3, our best models had the following model performance: XGBoost model resulted in a weighted recall of 0.9334011, weighted precision of 0.933738, and F1 score of 0.929060 and our best hypertuned logistic regression model had a weighted recall of 0.8865, weighted precision of 0.8919, and F1 of 0.8676.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data
# MAGIC 
# MAGIC We will be using domestic flight data from the US Department of Transportation and weather data from the National Oceanic and Atmospheric Administration, both from 2015-2021. We will also be using airport data that provides metadata on each airport from the US Department of Transportation. For exploratory purposes of Phase I, we will be using subsets of the flight and weather data from the first quarter of 2015. Subsequently, we will focus on the entire flight data departing from all major US airports for the 2015-2021 timeframe.
# MAGIC 
# MAGIC The flight data includes relevant information such as: 
# MAGIC - OntimeDeparture/ArrivalPercentage (percentage of flights that arrive and depart on time)
# MAGIC - Reporting_airline
# MAGIC - Departure/arrival date, times, and destinations details
# MAGIC - Departure/arrival performance
# MAGIC - Dep/ArrDelay (difference in minutes between scheduled and actual times)
# MAGIC - Dep/ArrDel15 (delay indicator 15 minutes or more 1=Yes)
# MAGIC - Flight Summaries (ActualElapsedTime, Airtime, etc)
# MAGIC - Causes of Delay (WeatherDelay, NASDelay, LateAircraftDelay)
# MAGIC 
# MAGIC The weather data includes relevant information such as: 
# MAGIC - Station ID
# MAGIC - Date of weather observations
# MAGIC - Visibility
# MAGIC - Mean/max of wind speed, wind gust
# MAGIC - Total precipitation/snow depth
# MAGIC - Fog, rain_drizzle, snow_ice_pellets, hail, thunder, tornado_funnel_cloud
# MAGIC 
# MAGIC The station dataset includes relevant information such as:
# MAGIC - Station ID
# MAGIC - ICAO code
# MAGIC 
# MAGIC The airport dataset includes relevant information such as:
# MAGIC - ICAO code
# MAGIC - IATA airport code

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary description of each table
# MAGIC 
# MAGIC ##### Flights
# MAGIC This dataset lists passenger flight's on-time performance data taken from the TranStats data collection available from the U.S. Department of Transportation (DOT) and contains flight information from 2015-2021. Reporting carriers are required to (or voluntarily) report on-time data for flights they operate: on-time arrival and departure data for non-stop domestic flights by month and year, by carrier and by origin and destination airport. The dataset includes scheduled and actual departure and arrival times, canceled and diverted flights, taxi-out and taxi-in times, causes of delay and cancellation, air time, and non-stop distance. This dataset will be used to track which flights were marked as delayed (departure) and collect information on the date, time, airline, departure city, etc. for each flight. 
# MAGIC 
# MAGIC ##### Weather
# MAGIC The weather data table includes weather data corresponding to the origin and destination airports at the time of departure and arrival respectively. It was downloaded from the National Oceanic and Atmospheric Administration repository and contains weather information from 2015 to 2021. The dataset contains summaries from major airport weather stations that include a daily account of temperature extremes, degree days, precipitation amounts and winds, along with hourly precipitation amounts and abbreviated 3-hourly weather observations. This dataset will be used in conjunction with the flight dataset to analyze the weather patterns on the days of delayed flights. 
# MAGIC 
# MAGIC ##### Station
# MAGIC The station data table provides metadata about each weather station, downloaded from the US Department of Transportation. It includes details such as the longitude/latitude of the weather station, the ICAO code, station identifier, etc. This dataset will be used with the flights and weather dataset to link the two datasets together. 
# MAGIC 
# MAGIC ##### Airport
# MAGIC 
# MAGIC The station data table provides metadata about each airport, downloaded from [Airport Database](https://www.partow.net/miscellaneous/airportdatabase/index.html#Downloads). It includes details such as the longitude/latitude of the airport, airport name, IATA airport code, ICAO code, etc. This dataset will be used with the station and weather datasets to link weather data to airport.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Raw Data

# COMMAND ----------

data_suffix = ''
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))
display(dbutils.fs.ls(f"{data_BASE_DIR}stations_data/"))
df_flights = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data{data_suffix}/")
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data{data_suffix}/")
df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Dictionary

# COMMAND ----------

display(df_flights)
df_flights.printSchema()

# COMMAND ----------

display(df_weather)
df_weather.printSchema()

# COMMAND ----------

display(df_stations)
df_stations.printSchema()

# COMMAND ----------

# https://www.partow.net/miscellaneous/airportdatabase/index.html#Downloads
# 01	ICAO Code	String (3-4 chars, A - Z)
# 02	IATA Code	String (3 chars, A - Z)
# 03	Airport Name	String
# 04	City/Town	String
# 05	Country	String
# 06	Latitude Degrees	Integer [0,360]
# 07	Latitude Minutes	Integer [0,60]
# 08	Latitude Seconds	Integer [0,60]
# 09	Latitude Direction	Char (N or S)
# 10	Longitude Degrees	Integer [0,360]
# 11	Longitude Minutes	Integer [0,60]
# 12	Longitude Seconds	Integer [0,60]
# 13	Longitude Direction	Char (E or W)
# 14	Altitude	Integer [-99999,+99999]
# (Altitude in meters from mean sea level)
# 16	Latitude Decimal Degrees	Floating point [-90,90]
# 17	Longitude Decimal Degrees	Floating point [-180,180]

airport_schema = T.StructType([
  T.StructField("call",T.StringType(), True),
  T.StructField("iata",T.StringType(), True),
  T.StructField("name",T.StringType(), True),
  T.StructField("city", T.StringType(), True),
  T.StructField("country", T.StringType(), True),     
  T.StructField("lat_deg", T.IntegerType(), False),
  T.StructField("lat_min", T.IntegerType(), False),
  T.StructField("lat_sec", T.IntegerType(), False),
  T.StructField("lat_dir", T.IntegerType(), False),
  T.StructField("lon_deg", T.IntegerType(), False),
  T.StructField("lon_min", T.IntegerType(), False),
  T.StructField("lon_sec", T.IntegerType(), False),
  T.StructField("lon_dir", T.IntegerType(), False),
  T.StructField("alt", T.IntegerType(), False),
  T.StructField("lat", T.DoubleType(), False),
  T.StructField("lon", T.DoubleType(), False),
])

df_airports = spark.read.format("csv") \
                   .option("header", False) \
                   .option("delimiter", ":") \
                   .schema(airport_schema) \
                   .load('dbfs:/FileStore/shared_uploads/jthsiao@berkeley.edu/GlobalAirportDatabase.txt')

display(df_airports)
display(df_airports.filter(df_airports['iata'] == 'SFO'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial EDA

# COMMAND ----------

# MAGIC %md
# MAGIC #### Dataset size (rows columns, train, test, validation)

# COMMAND ----------

print('Dimension of flights: ({}, {})'.format(df_flights.count(), len(df_flights.columns)))
print('Dimension of stations: ({}, {})'.format(df_stations.count(), len(df_stations.columns)))
print('Dimension of weather: ({}, {})'.format(df_weather.count(), len(df_weather.columns)))

# COMMAND ----------

df_flights.select('DEP_DELAY').describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing Values
# MAGIC 
# MAGIC The cell below provideds a count of the missing values found in each column.

# COMMAND ----------

from pyspark.sql.functions import col,isnan,when,count
df_Columns= df_flights
missing_value = df_flights.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_Columns.columns]
   ).toPandas()

# COMMAND ----------

df_flights.where(col("DEP_DELAY").isNull()).count()

# COMMAND ----------

cancelled = df_flights.where(df_flights['CANCELLED'] == 1).count()
diverted = df_flights.where(df_flights['DIVERTED'] == 1).count()
print(f'Percent of Cancelled Flights: {cancelled/df_flights.count()*100:.4f}%')
print(f'Percent of Diverted Flights: {diverted/df_flights.count()*100:.4f}%')

# COMMAND ----------

from matplotlib import pyplot as plt
df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
df_flights = df_airlines.toPandas()
#histogram of frequency of departure delays in minutes
fig, ax = plt.subplots(figsize=(10, 6))
df_flights.DEP_DELAY.plot(kind = 'hist', title = 'Flight Departure Delay in Minutes (actual - scheduled)', bins=50)
ax.set_xlabel("Minutes Delayed (Departures)", fontsize=12)
plt.xlim(0 , 400)
for rect in ax.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    space = 1
    label = "{:.0f}".format(y_value)
    ax.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom')
plt.show()

# COMMAND ----------

#comparison of departure/arrival flights that were classified as delayed or not (using 15 minute delay mark)
import pandas as pd

dep_del15 = df_flights['DEP_DEL15'].value_counts()
dep_del15 = pd.DataFrame(data=dep_del15)
dep_del15

# COMMAND ----------

#average number of delays per month in each quarter (January, February, and March)
df_flights.groupby('MONTH')['ARR_DELAY'].mean().plot(kind = 'bar', rot = 45, title = 'Avg Delay(in mins) by Month')

# COMMAND ----------

df_flights['DELAY'] = df_flights['DEP_DELAY']>15

# COMMAND ----------

#average minutes delayed by each day of the quarter
df_flights.groupby('FL_DATE')['DEP_DELAY'].mean().plot(title = 'Avg. Minutes Delayed by Day')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Machine Learning Algorithms
# MAGIC 
# MAGIC The models we will be using are supervised learning, classification algorithms, where the outcome will be either 1 (delay) or 0 (no delay). Based on research about commonly used models in the flight delay prediction industry, the below three models are chosen as the start point.
# MAGIC - Random Forest consists of various decision trees that select the suitable attribute for a node starting at the root and separate the data into subsets based on the selected attribute. It makes use of the bagging method and individual models of decision trees. The trained data are divided into random subsets, each with a decision tree. The data are given parallel to all trees in the forest, and the class that most trees predicted has the new data. We choose random forest as a starting point:
# MAGIC   - In A decision tree algorithm, Random Forests are less influenced by outliers than other algorithms.
# MAGIC   - They also do not make any assumptions about the underlying distribution of the data and can implicitly handle collinearity in features because if we have two highly similar features, the information gained from splitting on one of the features will also use up the predictive power of the other feature. 
# MAGIC   - Random Forests can be used for feature selection because if we fit the algorithm with features that are not useful, the algorithm simply won't use them to split the data. It's possible to extract the 'best' features.
# MAGIC 
# MAGIC - Logistic regression estimates the probability of an event occurring, such as voted or didn't vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. Logistic regression could be chosen to model flight delay for multiple reasons:
# MAGIC   - The weights of each feature trained by logistic regression are easily interpretable, as the sign of the weight indicates if a flight is more or less likely to be delayed if it has a high value for that feature. 
# MAGIC   - Logistic regression outputs a measure of confidence in its output through the probability of belonging to each class. This allows us to calculate the AUC measure, as opposed to if the only output was a label.
# MAGIC 
# MAGIC - Gradient Boosted Tree has only a single decision tree at the beginning representing the initial prediction for every training data. It uses a boosting method which means that individual models are trained sequentially. A tree is built, and its prediction is evaluated based on residual errors. Therefore, each tree model learns from mistakes made by the previous model. Building new trees will stop when an additional tree cannot improve the prediction. The data is given along a single root node tree. This model is considered one of our starting points as Gradient Boosted Decision Tree has shown great accuracy in modeling sequential data. With the help of this model, day-to-day sequences of the departure and arrival flight delays of an individual airport can be predicted efficiently.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outcome - Standard Metrics
# MAGIC 
# MAGIC \\(Accuracy = \frac{TP+TN}{TP+TN+FP+FN}\\)
# MAGIC 
# MAGIC \\(Precision = \frac{TP}{TP+FP}\\)
# MAGIC 
# MAGIC \\(Recall = \frac{TP}{TP+FN}\\)
# MAGIC 
# MAGIC \\(F1-Score = 2 \times \frac{Recall \times Precision}{Recall+Precision}\\)
# MAGIC 
# MAGIC The reason we are using precision and recall instead of accuracy is due to the possibility of imbalanced training data. While accuracy measures overall prediction accuracy, precision (actual positive/all predicted positives) and recall (all positive captured/all positive cases) are better measuring metrics for the model performance.
# MAGIC Domain-specific Metrics
# MAGIC We could use a confusion matrix to express the performance of our random forest model, the matrix expresses how many of a classifier’s predictions were correct, and when incorrect, where the classifier got confused.
# MAGIC 
# MAGIC ####Domain-specific Metrics
# MAGIC We could use a confusion matrix to express the performance of our random forest model, the matrix expresses how many of a classifier’s predictions were correct, and when incorrect, where the classifier got confused. <br>
# MAGIC 
# MAGIC The Area Under the ROC curve (AUC) will be used to evaluate how well our logistic regression model classifies positive and negative outcomes at all possible cutoffs. It can range from 0.5 to 1, and the larger it is the better. <br>
# MAGIC 
# MAGIC For the gradient boosted tree, we may have two loss functions, i.e: <br>
# MAGIC A custom loss function that we calculate the gradient for: \\(L(y_{i}, \hat{y_{i}})\\) <br>
# MAGIC The loss function used by the tree that fits the gradient, which is always squared loss. <br>
# MAGIC 
# MAGIC For the logistic regression model, L2 Regularization will be used to solve the problem of overfitting our model by penalizing the cost function. It does so by using an additional penalty term in the cost function.
# MAGIC \\(J\lgroup w \rgroup = \frac{1}{m} \displaystyle\sum\limits_{i=1}^m Cost(h(x^{i}), h(y^{i})) +\frac{\lambda}{2m}\displaystyle\sum\limits_{j=1}^n w_j^2\\)

# COMMAND ----------

img_BASE_DIR = "dbfs:/FileStore/shared_uploads/fidelianawar@berkeley.edu/Block_Diagram_W261_Phase_1.png"
display(dbutils.fs.ls(f"{img_BASE_DIR}"))

zero_df = spark.read.format("image").load(img_BASE_DIR)
display(zero_df)

# COMMAND ----------

pipeline_BASE_DIR = "dbfs:/FileStore/shared_uploads/fidelianawar@berkeley.edu/Pipeline_Steps_Phase_1.png"
display(dbutils.fs.ls(f"{img_BASE_DIR}"))

zero_df = spark.read.format("image").load(pipeline_BASE_DIR)
display(zero_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline Descriptions
# MAGIC 
# MAGIC ##### Feature Engineering
# MAGIC - Data Collection: Loading and aggregation from various sources.
# MAGIC - Pre-identified flight data from the US Department of Transportation contains flight information from 2015 to 2021.
# MAGIC - Pre-identified weather data from the National Oceanic and Atmospheric Administration and contains weather information from 2015 to 2021.
# MAGIC - Airport codes from Airport Codes (TXT) | US Department of Transportation.
# MAGIC - Data Evaluation: Profiling the data.
# MAGIC - Data Preparation: Cleaning, blending, wrangling.
# MAGIC 
# MAGIC ##### Modeling
# MAGIC - Model Training: Training with various methodologies via our composed dataset.
# MAGIC - Model Validation: Leveraging validation dataset to perform quick look evaluation of a trained model, using cross validation, etc.
# MAGIC - Model Metrics: Observing the training output,metrics generation and evaluation.
# MAGIC 
# MAGIC ##### Interpreting
# MAGIC - Test: Evaluate the trained model on the test dataset.
# MAGIC - Evaluation: Gain greater understanding of the relationship between features, classification performance metrics, identification of insights.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task Splitting
# MAGIC 
# MAGIC - Fidelia plans to build text-based features based on the product description and optimize random forest models using all available features.
# MAGIC - Jude might sign up for processing all the numerical data (standardizing, dealing with missing values) and optimizing regularized logistic regression models using all available features.
# MAGIC - Chi will select the appropriate features to optimize the gradient boosted tree models using all available features.
# MAGIC - Jerry is gonna do the data cleaning, and data wrangling to prepare the initial inputs for the models, and aggregate the final result of each model to evaluate the performance across our different approaches.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC 
# MAGIC During phase 1 of the project, the team clarified the project goals and performed initial exploratory data analysis and model selection. We were provided with three sets of useful data (flight data, weather data, and airport data) which all contain potentially relevant information for predicting flight delays, and we anticipate large effort data joints, feature engineering, and feature selection. Our modeling approach will be supervised learning algorithms to classify flights as delay( >15 min late) or no delay( <= 15 minutes late), and the team picked three modeling approaches that are commonly used in the industry today. In addition to using accuracy to evaluate model performance, precision, recall, and F-1 scores are more accurate measurements for this specific task due to the possibility of an unbalanced dataset. The data processing and model training will be done using Spark, and we’ve clearly laid out the pipeline steps. Finally, the team has agreed on a project timeline and roles and responsibilities for each team member.
# MAGIC 
# MAGIC ### Open Issues
# MAGIC Some issues we can foresee in this initial phase of the project are:
# MAGIC - Data cleaning - how do we handle null values
# MAGIC    - Flights
# MAGIC    - Weather
# MAGIC - Joining the datasets.
# MAGIC - Data formats especially when handling dates formatting
# MAGIC - Converted all times to UTC
# MAGIC - Feature selection - which are the most important features to keep
# MAGIC - Train-test split and cross-validation for time-series data
# MAGIC - Bias-variance tradeoff of the model
# MAGIC - Generalizability of the model
