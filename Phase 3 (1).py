# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3 - Summary
# MAGIC In this phase, the team worked on identifying new features to create a better predictive model, fine-tuning our pipeline using grid earch, implementing two different models with hyper parameter tuning and cross-validation: the XGBoost model and logistic regression model. We also analyzed differences in performance and performed experiments on all data for the new features and experimental settings and reported evaluation metrics over the dataset. One of the main highlights of this phase was using the PageRank analogy and graph theory lessons learned in class and applying them to our model. We used the PageRank method to rank airports because airplane flight paths form a node-link graph with each airport as a node and a flight in between as an edge. We analyzed our results from the two hyperparameter tuned models, including random 3 fold CV, custom fold CV, and XGBoost, and considered which features we want to optimize further for future iterations. 

# COMMAND ----------

from collections import Counter
import datetime
import os
import re

from pyspark.sql.functions import col
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import split

from pyspark.ml.feature import VectorAssembler, VectorIndexer, StandardScaler, PCA
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

from sparkdl.xgboost import XgboostRegressor
from imblearn.under_sampling import RandomUnderSampler 

import airporttime

# COMMAND ----------

blob_container = "w261accufly" # The name of your container created in https://portal.azure.com
storage_account = "w261accufly" # The name of your Storage account created in https://portal.azure.com
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
# MAGIC ## Load Saved DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unscaled

# COMMAND ----------

sample_suffix = '_under'
data_suffix = '' # _3m
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
df_flight_weather_p = spark.read.parquet(f"{blob_url}/df_flight_weather_p{data_suffix}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scaled

# COMMAND ----------

df_flight_final_feature = spark.read.parquet(f"{blob_url}/df_flight_final_feature{data_suffix}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC * Based on a fairly good model result in phase 2, we think we already have a good number of additional features that showed a fair amount of correlation with the class variation in EDA: season, holiday, part of the day, busy airport. 
# MAGIC * The only additional feature in phase 3 is the a graph based airport rank, which is done using PageRank algorithm on a node-linked graph where nodes are departure & destination airports and the links are the flights paths. This feature is also standardized using the Z score methodology
# MAGIC * In summary, the additional features can be grouped into two categories: 
# MAGIC    * Time based - intuitively, time of flight is a big driver of delay. Busy seasons cause shortage of gates, difficulty for traffic control and these can all contribute to flight delays. so we added several indicators for these time factors. Season: spring/summer/fall/winter, holiday: as some holidays vary on dates year over year, and in general Presidents day and MLK days don't contribute to busy travel as much as Christmas and 4th of July, we only indicated +- 1 day around major holidays such as Christmas Day, Thanksgiving Day, New Years Day and 4th of July. Day Part: the airport is busier certain times of the day than others, for example, mornings. So we segmented the 24 hours of the day into Day Part(early morning/morning/afternoon/evening/night). 
# MAGIC    * Airport based - Busy Airport: Airport Hubs has way more flight traffic than an airport in a smalltown, so we marked the busiest airports(ORD and ATL) and use that as an indicator as well. Airport Rank: Graph based airport rank

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Airport Rank
# MAGIC Since airplane flight path actually form a node-link graph, with each airport as a node and a flight in between as an edge. We thought it would be beneficial to use PageRank analogy and calculate the importance of each airport, and see if such rank score has any effect in predicting flight delay. 
# MAGIC Below, by transforming the flight departure iata and flight destination iata into a graph, the flight rank is calculated using 15 iterations and added to the dataset as an additional feature

# COMMAND ----------

#create RDD of the departure and destination airports
airportRDD = (df_flight_weather_p
              .groupBy('airport_iata','dest_iata')
              .agg(F.count('dest_iata').alias('no_flights'))
              .orderBy('airport_iata')
              .rdd
              .map(lambda x: (x[0],{x[1]:x[2]}))
              .reduceByKey(lambda x,y: {**x,**y})
              .cache()
             )

airportRDD.count()
#spark session
from pyspark.sql import SparkSession
import os

try:
    spark
except NameError:
    print('starting Spark')
    app_name = 'Lab6_notebook'
    master = "local[*]"
    spark = SparkSession\
            .builder\
            .appName(app_name)\
            .master(master)\
            .getOrCreate()
sc = spark.sparkContext

#help functions for PageRank
from pyspark.accumulators import AccumulatorParam
class FloatAccumulatorParam(AccumulatorParam):
  """
  Custom accumulator for use in page rank to keep track of various masses.

  IMPORTANT: accumulators should only be called inside actions to avoid duplication.
  We stringly recommend you use the 'foreach' action in your implementation below.
  """
  def zero(self, value):
      return value
  def addInPlace(self, val1, val2):
      return val1 + val2
def airportInit(airportRDD):
  nodeAccum = sc.accumulator(0.0, FloatAccumulatorParam())
  airportRDD.foreach(lambda x: nodeAccum.add(1))
  n = sc.broadcast(nodeAccum.value)
  airportInitRDD = airportRDD.mapValues(lambda x: (1/n.value,x)).cache()
  return airportInitRDD
  
def airportRank(airportInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    
    # initialize accumulators for total nodes
    nodeAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    def neighbor_redistribute(line):
            #there's no dangling node
            neighbor_dic = line[1][1]
            node_score = line[1][0]
            result = []
            if len(neighbor_dic) > 0:
                for neighbor,freq in neighbor_dic.items():
                    for i in range(freq):
                        result.append((neighbor, (node_score/sum(neighbor_dic.values()),{})))
                #don't forget adjacency list
                result.append((line[0],(0,neighbor_dic)))
            else:
                #score no need to be distributed
                result.append((line[0],(0,{})))
                #no need to add adj list
            for item in result:
                yield item

#     def dang_mass(line):
#         neighbor_list = ast.literal_eval(line[1][1])
#         node_score = line[1][0]
#         if len(neighbor_list) == 0:
#             mmAccum.add(node_score)
        
    def node_count(line):
        nodeAccum.add(1)

    def calc_p(v1,v2):
        """aggregate probability of nodes and adjacency list"""
        #calculate P
        return (v1[0]+v2[0],{**v1[1],**v2[1]})

    def score_update(x):
        score = x[0]
        new_score = a.value/N.value + d.value*score
        return x   

        
    # write your main Spark Job here (including the for loop to iterate)
    # for reference, the master solution is 21 lines including comments & whitespace
    
    curRDD = airportInitRDD
    for i in range(maxIter):
      nodeAccum.value = 0
      RDD1 = curRDD.flatMap(neighbor_redistribute).reduceByKey(calc_p).cache()
      RDD1.foreach(node_count)
      if verbose == True:
        print(nodeAccum.value)
      N = sc.broadcast(nodeAccum.value)
      curRDD = RDD1.mapValues(score_update).cache()
        
    steadyStateRDD = curRDD.mapValues(lambda x: x[0]).cache()
    
    return steadyStateRDD
  
airportInitRDD = airportInit(airportRDD)
airportRankRDD = airportRank(airportInitRDD, alpha = 0.15, maxIter = 15, verbose = False)
airportRankDF = airportRankRDD.toDF()

#rename columns for join
airportRankDF = (airportRankDF.withColumnRenamed('_1','dep_iata')
                     .withColumnRenamed('_2','dep_rank'))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Airport Rank Join 
# MAGIC (previous joined + airport rank)

# COMMAND ----------

#join rank with flight_weatehr_p
df_flight_weather_p = df_flight_weather_p.join(airportRankDF, df_flight_weather_p.airport_iata == airportRankDF.dep_iata,"left")

#write 2nd join to blob
#df_flight_weather_p.write.mode('overwrite').parquet(f"{blob_url}/df_flight_weather_rank{data_suffix}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### *** Final Table with Numeric Features UNSCALED ***

# COMMAND ----------

df_flight_weather_p = spark.read.parquet(f"{blob_url}/df_flight_weather_rank{data_suffix}").drop('dep_iata')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Standardize Numeric Columns including Page Rank
# MAGIC Since pagerank is also a numeric feature, we'd like to standardize it just as the other weather numeric features

# COMMAND ----------

# list out all the numeric columns
numeric_col = ['hourly_dry_bulb_temperature',
                    'hourly_precipitation',
                    'hourly_relative_humidity',
                    'hourly_visibility',
                    'hourly_wind_gust_speed',
                    'hourly_wind_speed',
                    'daily_precipitation',
                    'daily_snow_depth',
                    'daily_snow_fall',
                    'dep_rank']
# fill out the NA with 0
df_feature = df_flight_weather_p.fillna(0)

# standardize numeric columns
unlist = udf(lambda x: round(float(list(x)[0]),2), T.DoubleType())
for c in numeric_col:
  assembler = VectorAssembler(inputCols = [c],outputCol = c + '_Vect')
  scaler = StandardScaler(inputCol = c + '_Vect', 
    outputCol = c + '_Scaled',
    withMean = True,
    withStd = True
)
  pipeline = Pipeline(stages = [assembler,scaler])
  
  df_feature = (pipeline.fit(df_feature).transform(df_feature)
       .withColumn(c+'_scaled',unlist(c+'_Scaled')).drop(c+'_Vect',c))

#df_feature.write.mode('overwrite').parquet(f"{blob_url}/df_flight_final_feature{data_suffix}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### *** Final Table with Numeric Columns SCALED ***

# COMMAND ----------

df_flight_final_feature = spark.read.parquet(f"{blob_url}/df_flight_final_feature{data_suffix}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualization and Sampling

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA on Airport Rank Join
# MAGIC For some features such as hourly_precipitation, hourly_visibility, hourly_wind_speed and etc it seems like the distribution is centered around 0, which might be due to the data imputation

# COMMAND ----------

scaled_col = [c+'_scaled' for c in numeric_col]
feature_pd = (df_flight_final_feature.select(*scaled_col).sample(False,0.01,81)).toPandas()
import matplotlib.pyplot as plt
#fig, ax = plt.subplots(2,5, figsize=(20,10))
fig=plt.figure(figsize=(20,10))
for i, c in zip(range(5),scaled_col[:5]):
    ax=fig.add_subplot(2,5,i+1)
    feature_pd[c].hist(bins=10,ax=ax)
    ax.set_title(c)
fig.tight_layout()  # Improves appearance a bit.
plt.show()
fig=plt.figure(figsize=(20,10))
for i, c in zip(range(5),scaled_col[5:]):
    ax=fig.add_subplot(2,5,i+1)
    feature_pd[c].hist(bins=10,ax=ax)
    ax.set_title(c)
fig.tight_layout()  # Improves appearance a bit.
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA on Airport Rank Join
# MAGIC With the added airport rank feature, it's worthwhile to re-run the PCA analysis to see if any significant changes to the principle components

# COMMAND ----------

#selecting columns of features to be used in model training
feature_col = ['cancelled','diverted','carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay','holiday',
 'spring','summer','fall','winter','early_morning','morning','afternoon','evening','night','busy_airport'] + [c+'_scaled' for c in numeric_col]+['delay_label']

feature_table = df_flight_final_feature.select([col for col in feature_col if col != 'delay_label'])
#PCA Analysis
from pyspark.ml.linalg import Vectors, VectorUDT
#vectorize PCA
assembler = VectorAssembler(inputCols=feature_col[:-1], outputCol="features")
assembler.transform(feature_table).select('features')
pca_vec = (feature_table .select(
  udf(Vectors.dense, VectorUDT())(*feature_table.columns)
).toDF("features"))

#choose
n_components = 5
pca = PCA(
    k = n_components, 
    inputCol = 'features', 
    outputCol = 'pcaFeatures'
).fit(pca_vec)

df_pca = pca.transform(pca_vec)
print('Explained Variance Ratio', pca.explainedVariance.toArray())
df_pca.show(6)
#pca_x and pca_y will be used for ML, need to split first
import numpy as np
pca_x = df_pca.select('pcaFeatures').rdd.map(lambda row: row[0]).collect()
pca_x = np.array(pca_x).T
pca_x
pca_y = feature_table.select('delay_label').collect()
pca_y = np.array(pca_y.T)
pca_x.shape
pca_y.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Split (Train/Test Sets)

# COMMAND ----------

df_feature = df_flight_final_feature
year = udf(lambda d: d.year, T.IntegerType())
year_order = udf(lambda d: d.year%100-15, T.IntegerType())
year_quarter = udf(lambda d,q: d.year*10+q, T.IntegerType())
df_final = df_feature.withColumn('year', year(df_feature.target_dep_utc_timestamp))\
                     .withColumn('year_order', year_order(df_feature.target_dep_utc_timestamp))\
                     .withColumn('year_quarter', year_quarter(df_feature.target_dep_utc_timestamp, df_feature.quarter))

df_test = df_final.filter(col('year') == 2021)
df_train = df_final.filter(col('year') < 2021)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Undersample

# COMMAND ----------

# https://stackoverflow.com/questions/53978683/how-to-undersampling-the-majority-class-using-pyspark
def resample(base_features,ratio,class_field,base_class,stat=False):
    pos = base_features.filter(col(class_field)==base_class)
    neg = base_features.filter(col(class_field)!=base_class)
    total_pos = pos.count()
    total_neg = neg.count()
    if stat:
      print(f'pos: {total_pos}')
      print(f'neg: {total_neg}')
      return base_features
    fraction=float(total_pos*ratio)/float(total_neg)
    sampled = neg.sample(False,fraction)
    return sampled.union(pos)
  
df_final_test = resample(df_test, 1, 'delay_label', 1)
df_final_train = resample(df_train, 1, 'delay_label', 1)
resample(df_final_train, 1, 'delay_label', 1, stat=True)
resample(df_final_test, 1, 'delay_label', 1, stat=True)

# COMMAND ----------

# select the meaningful independent variables and dependent variable
features = ['year_order', 'quarter', 'month', 'day_of_week','carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay','holiday','spring','summer', 'fall','winter','early_morning','morning','afternoon','evening','night','busy_airport','hourly_dry_bulb_temperature_scaled','hourly_precipitation_scaled','hourly_relative_humidity_scaled','hourly_visibility_scaled','hourly_wind_gust_speed_scaled', 'hourly_wind_speed_scaled','daily_precipitation_scaled','daily_snow_depth_scaled','daily_snow_fall_scaled', 'dep_rank_scaled']
label = ["delay_label"]
df_final_test = df_final_test.select(*features, *label).cache()
df_final_train = df_final_train.select(*features, *label).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### *** Get Final Feature Scaled Table for ML ***

# COMMAND ----------

df_flight_final_feature = spark.read.parquet(f"{blob_url}/df_flight_final_feature{data_suffix}")

# COMMAND ----------

# vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=features,outputCol="rawFeatures")
 
# vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features")

xgb_regressor = XgboostRegressor(num_workers=10, labelCol="delay_label", missing=0.0)

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
 
os.environ["PYSPARK_PIN_THREAD"] = "true"

# Define a grid of hyperparameters to test:
#  - maxDepth: maximum depth of each decision tree 
#  - maxIter: iterations, or the total number of trees 
paramGrid = ParamGridBuilder()\
  .addGrid(xgb_regressor.max_depth, [2, 5])\
  .addGrid(xgb_regressor.n_estimators, [10, 100])\
  .build()
 
# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol=xgb_regressor.getLabelCol(),
                                predictionCol=xgb_regressor.getPredictionCol())
 


# COMMAND ----------

# MAGIC %md
# MAGIC #### Random 3 Fold CV

# COMMAND ----------

# Declare the CrossValidator, which performs the model tuning.
cv = CrossValidator(estimator=xgb_regressor, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Custom Fold CV by Year

# COMMAND ----------

spark.sparkContext.addPyFile("dbfs:/custom_cv_jth.py")
import custom_cv_jth

cv = custom_cv_jth.CustomCrossValidator(estimator=xgb_regressor, evaluator=evaluator, estimatorParamMaps=paramGrid, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Custom Fold CV by Rolling Year

# COMMAND ----------

spark.sparkContext.addPyFile("dbfs:/custom_cv_jth1.py")
import custom_cv_jth1

cv = custom_cv_jth1.CustomCrossValidator(estimator=xgb_regressor, evaluator=evaluator, estimatorParamMaps=paramGrid, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])
# pipeline = Pipeline(stages=[vectorIndexer, cv])

# COMMAND ----------

pipelineModel = pipeline.fit(df_final_train)

# COMMAND ----------

predictions = pipelineModel.transform(df_final_test)

# COMMAND ----------

display(predictions.select("delay_label", "prediction", *features))

# COMMAND ----------

rmse = evaluator.evaluate(predictions)
print("RMSE on our test set: %g" % rmse)

# COMMAND ----------

# establise the evaluators
recall = MulticlassClassificationEvaluator(metricName="weightedRecall", labelCol="delay_label", predictionCol="pred_adjust")
precision = MulticlassClassificationEvaluator(metricName="weightedPrecision", labelCol="delay_label", predictionCol="pred_adjust")
f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="delay_label", predictionCol="pred_adjust")

# COMMAND ----------

df_pred = predictions.withColumn('pred_adjust', F.round(predictions.prediction))

# COMMAND ----------

# evaluation metics for test data
print(f"Weighted Recall: {recall.evaluate(df_pred)}")
print(f"Weighted Precision: {precision.evaluate(df_pred)}")
print(f"F1: {f1.evaluate(df_pred)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter Tuning
# MAGIC 
# MAGIC #### XGBoost
# MAGIC Limited preliminary hyperparameter tuning to the following due to time constraints. No regularization parameters hypertuned.  Next phase planning to tune gamma.  Using default loss function regression with squared loss.
# MAGIC * maxDepth: maximum depth of each decision tree 
# MAGIC    * [2, 5]
# MAGIC    * Stayed below the default setting of 6 due to overfitting concerns.
# MAGIC * n_estimators: the total number of trees 
# MAGIC    * [10, 100]
# MAGIC    * Limited due to time constraints

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modeling Pipelines (XGBoost Updated)
# MAGIC 
# MAGIC 
# MAGIC * StandardScaler (Z Score)
# MAGIC * Train / Test Split (2015-2020 / 2021)
# MAGIC * Undersample Train / Test
# MAGIC * VectorAssembler
# MAGIC * VectorIndexer
# MAGIC * CrossValidation (By year) Experiment 4
# MAGIC   * Grid Search
# MAGIC       * max depth [2,5]
# MAGIC       * num of estimators [10, 100]
# MAGIC   * RegressionEvaluator (RMSE)
# MAGIC   * CV (train / test) [2015 / 2016, 2016 / 2017, 2017 / 2018, 2018 / 2019, 2020 / 2025]
# MAGIC   * Best Model:  {Param(parent='XgboostRegressor_582ec9dd466f', name='max_depth', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param max_depth'): 5, Param(parent='XgboostRegressor_582ec9dd466f', name='n_estimators', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param n_estimators'): 10} Detailed Score [0.40017360073686226, 0.34017707327754027, 0.34893689360364577, 0.3594403817919477, 0.37539452923320793, 0.42425764976522906] Avg Score 0.37473002140140554
# MAGIC * CrossValidation (By year rolling) Experiment 5
# MAGIC   * Grid Search
# MAGIC       * max depth [2,5]
# MAGIC       * num of estimators [10, 100]
# MAGIC   * RegressionEvaluator (RMSE)
# MAGIC   * CV (train / test) 
# MAGIC     * 2015 train + 2016 test
# MAGIC     * 2015, 2016 train + 2017 test
# MAGIC     * 2015, 2016,2017 train + 2018 test
# MAGIC     * 2015, 2016,2017, 2018 train + 2019 test
# MAGIC     * 2015, 2016,2017, 2018, 2019 train + 2020 test
# MAGIC   * Best Model:  {Param(parent='XgboostRegressor_ec007ca2f653', name='max_depth', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param max_depth'): 5, Param(parent='XgboostRegressor_ec007ca2f653', name='n_estimators', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param n_estimators'): 10} Detailed Score [0.4125298293056656, 0.344710934255943, 0.3448790848413747, 0.35832744833980107, 0.3505780090641942] Avg Score 0.3622050611613957
# MAGIC * CrossValidation (Random 3 Fold) Experiment 1, 2, 3
# MAGIC   * Grid Search
# MAGIC       * max depth [2,5]
# MAGIC       * num of estimators [10, 100]
# MAGIC   * RegressionEvaluator (RMSE)
# MAGIC   * CV (train / test) Random
# MAGIC * Fit train
# MAGIC * Transform test
# MAGIC * Evaluate
# MAGIC   * RMSE, F1, Weighted Recall, Weighted Precision

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Pipeline, Pipeline Model, Data

# COMMAND ----------

# MAGIC %sh 
# MAGIC rm -r dbfs:/xgboost/pipeline_
# MAGIC rm -r dbfs:/xgboost/pipeline_model_

# COMMAND ----------

dbutils.fs.ls("/")
# base_name = "xgboost_cv_rank"
# base_name = "xgboost_cv_rank_full"
base_name = "xgboost_cvcus_rank_full"
data_suffix = ""
sample_suffix = "_under"
print(f"Tag: {base_name}{data_suffix}{sample_suffix}")

# COMMAND ----------

df_final_train.write.parquet(f"{blob_url}/df_{base_name}_train{data_suffix}{sample_suffix}")
df_final_test.write.parquet(f"{blob_url}/df_{base_name}_test{data_suffix}{sample_suffix}")

# COMMAND ----------

# Save the pipeline that created the model
pipeline.save(f"/team16/{base_name}/pipeline{data_suffix}{sample_suffix}")
 
# Save the model itself
pipelineModel.save(f"/team16/{base_name}/pipeline_model{data_suffix}{sample_suffix}")

# COMMAND ----------

df_pred.write.parquet(f"{blob_url}/df_{base_name}_pred{data_suffix}{sample_suffix}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load

# COMMAND ----------

display(dbutils.fs.ls("/team16/xgboost"))
base_name = "xgboost"
data_suffix = ""
sample_suffix = "_under"
print(f"Tag: {base_name}{data_suffix}{sample_suffix}")

# COMMAND ----------

# df_final_train = spark.read.parquet(f"{blob_url}/df_{base_name}_train{data_suffix}{sample_suffix}")
# df_final_test = spark.read.parquet(f"{blob_url}/df_{base_name}_test{data_suffix}{sample_suffix}")

loaded_pipeline = Pipeline.load(f"/team16/{base_name}/pipeline{data_suffix}{sample_suffix}")
loaded_pipelineModel = PipelineModel.load(f"/team16/{base_name}/pipeline_model{data_suffix}{sample_suffix}")

df_pred = spark.read.parquet(f"{blob_url}/df_{base_name}{data_suffix}{sample_suffix}")


# COMMAND ----------

display(df_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiments (Logistic Regression) on ALL the data for new features and experimental settings

# COMMAND ----------

# MAGIC %md
# MAGIC This experiement will be based on the previous logistic regression basic model in Phase 2, we are aiming to improve <br>
# MAGIC the performance of this model by using CV on time series, hypertuning, logistic regression category feature index.

# COMMAND ----------

df_flight_final_feature = spark.read.parquet(f"{blob_url}/df_flight_final_feature{data_suffix}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preparation for the Experiment

# COMMAND ----------

from pyspark.sql.functions import split
from pyspark.sql.types import IntegerType
df_final =  df_flight_final_feature.withColumn('year', split(df_flight_final_feature['fl_date'], '-').getItem(0).cast(IntegerType()))
df_final_test = df_final.filter(col('year') == 2021)
df_final_train = df_final.filter(col('year') < 2021)

# COMMAND ----------

# select the meaningful independent variables and dependent variable
features = ['quarter', 'month', 'day_of_week', 'year', 'carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay','holiday','spring','summer', 'fall','winter','early_morning','morning','afternoon','evening','night','busy_airport','hourly_dry_bulb_temperature_scaled','hourly_precipitation_scaled','hourly_relative_humidity_scaled','hourly_visibility_scaled','hourly_wind_gust_speed_scaled', 'hourly_wind_speed_scaled','daily_precipitation_scaled','daily_snow_depth_scaled','daily_snow_fall_scaled', 'dep_rank_scaled']
label = ["delay_label"]
df_final_test =  df_final_test.select(*features, *label)
df_final_train =  df_final_train.select(*features, *label)

# COMMAND ----------

# create dictionary of dataframes for custom cv fn to loop through
# assign train and test based on time series split 

d = {}

d['df1'] = df_final_train.filter(df_final_train.year <= 2016)\
                   .withColumn('cv', F.when(df_final_train.year == 2015, 'train')
                                         .otherwise('test'))

d['df2'] = df_final_train.filter(df_final_train.year <= 2017)\
                   .withColumn('cv', F.when(df_final_train.year <= 2016, 'train')
                                         .otherwise('test'))

d['df3'] = df_final_train.filter(df_final_train.year <= 2018)\
                   .withColumn('cv', F.when(df_final_train.year <= 2017, 'train')
                                         .otherwise('test'))

d['df4'] = df_final_train.filter(df_final_train.year <= 2019)\
                   .withColumn('cv', F.when(df_final_train.year <= 2018, 'train')
                                         .otherwise('test'))

d['df5'] = df_final_train.filter(df_final_train.year <= 2020)\
                   .withColumn('cv', F.when(df_final_train.year <= 2019, 'train')
                                         .otherwise('test'))

# COMMAND ----------

categoricals = ['quarter', 'month', 'day_of_week', 'holiday','year','spring','summer', 'fall','winter','early_morning','morning','afternoon','evening','night','busy_airport']
numerics =['carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay','hourly_dry_bulb_temperature_scaled','hourly_precipitation_scaled','hourly_relative_humidity_scaled','hourly_visibility_scaled','hourly_wind_gust_speed_scaled' ,'hourly_wind_speed_scaled','daily_precipitation_scaled','daily_snow_depth_scaled','daily_snow_fall_scaled', 'dep_rank_scaled']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression Modeling with custom-validation & hyper-tuning

# COMMAND ----------

# import custom cv module
spark.sparkContext.addPyFile("dbfs:/custom_cv.py")
from custom_cv import CustomCrossValidator
from pyspark.ml.feature import StandardScaler, Imputer
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder

evaluator = BinaryClassificationEvaluator()

## Current possible ways to handle categoricals in string indexer is 'error', 'keep', and 'skip'
indexers = map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), categoricals)
ohes = map(lambda c: OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class"),categoricals)
imputers = Imputer(inputCols = numerics, outputCols = numerics)

# Establish features columns
featureCols = list(map(lambda c: c+"_class", categoricals)) + numerics

# Build the stage for the ML pipeline
# Build the stage for the ML pipeline
model_matrix_stages = list(indexers) + list(ohes) + [imputers] + \
                     [VectorAssembler(inputCols=featureCols, outputCol="features"), StringIndexer(inputCol="delay_label", outputCol="label")]

# Use logistic regression 
lr = LogisticRegression(featuresCol = "features")

# Build our ML pipeline
pipeline = Pipeline(stages=model_matrix_stages + [lr])

# Build the parameter grid for model tuning
grid = ParamGridBuilder() \
              .addGrid(lr.regParam, [0.1, 0.01]) \
              .build()

# COMMAND ----------

cv = CustomCrossValidator(estimator=pipeline, 
                          estimatorParamMaps=grid, 
                          evaluator=evaluator,
                          splitWord = ('train', 'test'), 
                          cvCol = 'cv', 
                          parallelism=4)

cvModel = cv.fit(d)
prediction = cvModel.transform(df_final_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Result Evaluation

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

recall = MulticlassClassificationEvaluator(metricName="weightedRecall", labelCol="delay_label")
precision = MulticlassClassificationEvaluator(metricName="weightedPrecision", labelCol="delay_label")
f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="delay_label")

print(f"Weighted Recall: {recall.evaluate(prediction)}")
print(f"Weighted Precision: {precision.evaluate(prediction)}")
print(f"F1: {f1.evaluate(prediction)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gap Analysis
# MAGIC Based on the current scoreboard (Mon 7/25 11:45pm PST), the results of our updated models seems to be scoring well compared to our peers. Our precision, recall, and F1 scores for this phase are higher than all except one of our peers. We also see improvement in our metrics from last phase's evaluation. For this phase, we believe we have achieved reasonable metrics after our hyper parameter tuning and have a few points of consideration to improve our scores even further for future phases. Based on the fishbone diagram, we have four potential causes of “low” metrics that we can try to improve for next phase:
# MAGIC 
# MAGIC - <b>Feature Selection Techniques</b>: While we believe we identified the most relevant features for the model and trained on the meaningful independent/dependent variables, there is no guarantee that the features we selected were the “best” ones. It could be possible that we do not provide the model with enough features (or too many features), which would affect the final performance. In this scenario, we could use further PCA techniques to aid in choose the best features to ensure our model provides the most useful information. We also did not leverage features like COVID data or the previous flight delay tail attribute, which we can incorporate for future iterations.
# MAGIC - <b>Data Analysis Decisions</b>: We are using z-score instead of min-max and standard scalar normalization. Min-max normalization guarantees that all features will have the exact same scale but does not handle outliers well, whereas z-score normalization handles outliers, but does not produce normalized data with the exact same scale. It would be interesting to explore the difference in evaluation for the models in future phases and compare with the performance of previous phases. We also did not use PCA for this phase because of time constraints, so we will be trying PCA for future phases to increase our model performance.
# MAGIC - <b>Limited Performance Tuning</b> Because the models take a decent amount of time to train/fit, it makes it difficult to be able to change many parameters frequently to analyze the results. To improve for this in the next phase, we can test and document the performance of different parameter changes overtime and keep the one with the highest results.
# MAGIC - <b>Algorithm Methodology</b>: For this phase, our CV hyper tuned model used time series cross validation on a rolling basis over the years, but we would be curious to explore if a better solution exists if it were modeled over months or quarters. For the models, we also only tuned the regression hyper parameter, so it would be interesting to explore if our metrics improve if we tune more parameters, such as gamma.

# COMMAND ----------

fishbone_BASE_DIR = "dbfs:/FileStore/shared_uploads/fidelianawar@berkeley.edu/Gap_Analysis_Fishbone_Diagram.png"
display(dbutils.fs.ls(f"{fishbone_BASE_DIR}"))

zero_df = spark.read.format("image").load(fishbone_BASE_DIR)
display(zero_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### XGBoost Results
# MAGIC 
# MAGIC For Experiment details see Pipeline Models Section under CrossValidation which is where the majority of modifications for experiements were made.
# MAGIC 
# MAGIC | Experiment | Sample | Popularity | Features | Folds | Weighted Recall | Weighted Precision | F1       | RMSE     |
# MAGIC |-------|--------|------------|----------|-------|-----------------|--------------------|----------|----------|
# MAGIC | 1 | Under  | No         | Full     | 3 (random)    | 0.8404057       | 0.871267           | 0.837105 | 0.358528 |
# MAGIC | 2 | Under  | Yes        | Reduced  | 3 (random)     | 0.8172313       | 0.850636           | 0.735041 | 0.386793 |
# MAGIC | 3 | Under  | Yes        | Full  | 3 (random)     | 0.9334011  | 0.933738 | 0.929060 |  0.24907 |
# MAGIC | 4 | Under  | Yes        | Full  | 6 (year) | 0.361338   | 0.838923 | 0.8707551 | 0.835407 |
# MAGIC | 5 | Under  | Yes        | Full  | 6 (year) | 0.423011   | 0.505587 | 0.549779 | 0.363272 |
# MAGIC | 6 | Under  | Yes        | Full  |  3 (random) | 0.410261   | 0.674668 | 0.690508 | 0.6677294 |
# MAGIC 
# MAGIC * Sample: Undersampling
# MAGIC * Popularity: Using PageRanked Airport Popularity
# MAGIC * Features: Set of features selected
# MAGIC    * Reduced: ['quarter', 'month', 'day_of_week', 'holiday','spring','summer', 'fall','winter','early_morning','morning','afternoon','evening','night','busy_airport','hourly_dry_bulb_temperature_scaled','hourly_precipitation_scaled','hourly_relative_humidity_scaled','hourly_visibility_scaled','hourly_wind_gust_speed_scaled', 'hourly_wind_speed_scaled','daily_precipitation_scaled','daily_snow_depth_scaled','daily_snow_fall_scaled', 'dep_rank_scaled']
# MAGIC    * Full: ['quarter', 'month', 'day_of_week','carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay','holiday','spring','summer', 'fall','winter','early_morning','morning','afternoon','evening','night','busy_airport','hourly_dry_bulb_temperature_scaled','hourly_precipitation_scaled','hourly_relative_humidity_scaled','hourly_visibility_scaled','hourly_wind_gust_speed_scaled', 'hourly_wind_speed_scaled','daily_precipitation_scaled','daily_snow_depth_scaled','daily_snow_fall_scaled', 'dep_rank_scaled'] 
# MAGIC * Folds: Number of folds and method of selection
# MAGIC * Recall: Metric
# MAGIC * Precision: Metric
# MAGIC * F1: Metric
# MAGIC * RMSE: Metric
# MAGIC 
# MAGIC ##### Experiement Parameters
# MAGIC Experiment 1,2,3:
# MAGIC ```
# MAGIC {Param(parent='XgboostRegressor_582ec9dd466f', name='max_depth', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param max_depth'): 5, 
# MAGIC  Param(parent='XgboostRegressor_582ec9dd466f', name='n_estimators', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param n_estimators'): 10} 
# MAGIC ```
# MAGIC 
# MAGIC Experiment 4:
# MAGIC ```
# MAGIC {Param(parent='XgboostRegressor_582ec9dd466f', name='max_depth', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param max_depth'): 5, 
# MAGIC  Param(parent='XgboostRegressor_582ec9dd466f', name='n_estimators', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param n_estimators'): 10} 
# MAGIC  Detailed Score [0.40017360073686226, 0.34017707327754027, 0.34893689360364577, 0.3594403817919477, 0.37539452923320793, 0.42425764976522906] Avg Score 0.37473002140140554
# MAGIC ```
# MAGIC 
# MAGIC Experiment 5:
# MAGIC ```
# MAGIC {Param(parent='XgboostRegressor_ec007ca2f653', name='max_depth', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param max_depth'): 5, 
# MAGIC  Param(parent='XgboostRegressor_ec007ca2f653', name='n_estimators', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param n_estimators'): 10} 
# MAGIC  Detailed Score [0.4125298293056656, 0.344710934255943, 0.3448790848413747, 0.35832744833980107, 0.3505780090641942] Avg Score 0.3622050611613957
# MAGIC ```
# MAGIC 
# MAGIC Experiment 6:
# MAGIC ```
# MAGIC Out[23]: {Param(parent='XgboostRegressor_c4a620b7bd85', name='max_depth', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param max_depth'): 15,
# MAGIC  Param(parent='XgboostRegressor_c4a620b7bd85', name='n_estimators', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param n_estimators'): 50,
# MAGIC  Param(parent='XgboostRegressor_c4a620b7bd85', name='gamma', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param gamma'): 0.1,
# MAGIC  Param(parent='XgboostRegressor_c4a620b7bd85', name='objective', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param objective'): 'reg:squarederror',
# MAGIC  Param(parent='XgboostRegressor_c4a620b7bd85', name='learning_rate', doc='Refer to XGBoost doc of xgboost.sklearn.XGBRegressor for this param learning_rate'): 0.1}
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression Results
# MAGIC 
# MAGIC Our final experimental model is a logistic regression model built with hypertuning and CV on time series.
# MAGIC The cross validation we used is a rolling basis over the years, e.g. <br>
# MAGIC 2015 train + 2016 test<br>
# MAGIC 2015, 2016 train + 2017 test<br>
# MAGIC 2015, 2016,2017 train + 2018 test<br>
# MAGIC 2015, 2016,2017, 2018 train + 2019 test<br>
# MAGIC 2015, 2016,2017, 2018, 2019 train + 2020 test<br>
# MAGIC 
# MAGIC | Sample | Popularity | Features |Weighted Recall | Weighted Precision | F1       |
# MAGIC |--------|------------|----------|-----------------|--------------------|----------|
# MAGIC | Under  | Yes        | Full  |0.8865   |0.8919  | 0.8676 |
# MAGIC 
# MAGIC 
# MAGIC * Sample: no
# MAGIC * Popularity: Using PageRanked Airport Popularity
# MAGIC * Features: Set of features selected
# MAGIC    * Full: ['quarter', 'month', 'day_of_week','carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay','holiday','spring','summer', 'fall','winter','early_morning','morning','afternoon','evening','night','busy_airport','hourly_dry_bulb_temperature_scaled','hourly_precipitation_scaled','hourly_relative_humidity_scaled','hourly_visibility_scaled','hourly_wind_gust_speed_scaled', 'hourly_wind_speed_scaled','daily_precipitation_scaled','daily_snow_depth_scaled','daily_snow_fall_scaled', 'dep_rank_scaled'] 
# MAGIC * Recall: Metric
# MAGIC * Precision: Metric
# MAGIC * F1: Metric

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis
# MAGIC 
# MAGIC Looking at the tables above, we see the output of our hypertuned XGBoost and logistic regression models. Our best XGBoost model resulted in a weighted recall of 0.9334011, weighted precision of 0.933738, and F1 score of 0.929060 and our best hypertuned logistic regression model had a weighted recall of 0.8865, weighted precision of 0.8919, and F1 of 0.8676. Comparing these two models with each other, the XGBoost with 3 folds seems to have performed better. This could be due to the number of folds being fewer vs being done on a rolling basis over the years for the LR model. Reading the results tables in more detail, we can see how the inclusion of all features results in higher metrics than testing on the reduced datasets, so we know that we are on track with appropriately selected features based on model performance. Overall we are fairly confident that these models are performing similarly on unseen data as they do on training data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC 
# MAGIC The focus of the project is to build a predictive model that can accurately predict flight delays up to 2 hours in advance. The focus of this phase was to build upon our implementation from last phase to create models with more optimized parameters. We predict that machine learning pipelines with cutom features can accurately predict flight delays using the datasets provided because we are able to analyze past trends in weather conditions, reasons for delays, airports where there are more delays, and many other necessarily details combined to feed into our model to make predictions off of. We believe our model especially is able to accomplish this task based on the following:
# MAGIC 
# MAGIC - Feature engineering techniques performed
# MAGIC - Using PCA analysis to extract the most relevant features for our model
# MAGIC - Using PageRank to calculate the importance of each airport, and see if their rank score has any effect in predicting flight delays
# MAGIC - Performing hyper parameter tuning and cross validation over time series for our XGBoost and logistic regression models 
# MAGIC - Finding the most optimal features and parameters to continue to further improve our evaluation metrics
# MAGIC 
# MAGIC As we saw in our results, the performance of our models is pretty successful and accurate at predicting delays based on our evaluation metrics. We are satisfied with our feature selection choices and parameter optimizations for this phase, but still have other aspects we want to explore, including tuning more parameters in addition to regression or changing the cross validation time series to be over quarters/months. For phase IV and V, we want to work on implementing more sophisticated models such as Random Forests, choosing optimized hyperparameters, and pursue further findings such as multitask loss functions.
# MAGIC 
# MAGIC ### Improvements Based on Presentation Feedback
# MAGIC * Added histograms of the standardized numeric features.  For some features such as hourly_precipitation, hourly_visibility, hourly_wind_speed and etc it seems like the distribution is centered around 0, which might be due to the data imputation
# MAGIC * Based on feedback from class, we addressed the problem of an imbalanced dataset by adding undersampling to create a more balanced dataset which is subsequently used in XGBoost modeling.
