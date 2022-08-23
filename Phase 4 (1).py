# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4 Notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC This notebok contains the final iteration of feature engineering and model tuning based on feedback from phase 3. During this phase, the team realized some features should not be included because they contain information not known 2 hours before flight departure. The team also fixed some feature labeling to increase predictability. Since the flight dataset is imbalanced, undersampling was performed to create more balanced data for model training and testing. And finally, additional parameters are experimented to further tune the model.

# COMMAND ----------

# import the necessary libraries/paths/help functions
from collections import Counter
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler 

from pyspark.sql import Row
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import col
from pyspark.sql.functions import split

from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StandardScaler, PCA
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, MinMaxScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from sparkdl.xgboost import XgboostRegressor
from sklearn import neighbors

import airporttime

# import custom cv module
spark.sparkContext.addPyFile("dbfs:/custom_cv.py")
from custom_cv import CustomCrossValidator

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dropdown Parameter Support

# COMMAND ----------

# Enable dropdown parameter selection support
dbutils.widgets.dropdown("dataset", "scaled", 
                         ["prescaled", "scaled"])
dbutils.widgets.dropdown("eda", "No", 
                         ["Yes", "No"])
dbutils.widgets.dropdown("model_selection", "All", 
                         ["LogisticRegression", "RandomForrest", "TreeEnsembles", "All"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Storage Access and Parameter Loading

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

sc = spark.sparkContext

data_suffix = 'None'
data_suffix = '' if 'None' == data_suffix else data_suffix
dataset = dbutils.widgets.get('dataset')
eda = dbutils.widgets.get('eda')
model_selection = dbutils.widgets.get('model_selection')
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def plot_roc_curve(prediction):
    """
    utility function for plotting ROC curve
    """
    from sklearn.metrics import auc, roc_curve
    pred_pd = prediction.select(['delay_label', 'prediction', 'probability']).toPandas()
 
    pred_pd['probability'] = pred_pd['probability'].map(lambda x: list(x))
    pred_pd['encoded_label'] = pred_pd['delay_label'].map(lambda x: np.eye(2)[int(x)])
    
    y_pred = np.array(pred_pd['probability'].tolist())
    y_true = np.array(pred_pd['encoded_label'].tolist())
 
    fpr, tpr, threshold = roc_curve(y_score=y_pred[:,0], y_true=y_true[:,0])
    auc_val = auc(fpr, tpr)
 
    plt.figure()
    plt.plot([0,1], [0,1], '--', color='green')
    plt.plot(fpr, tpr, label='auc = {:.3f}'.format(auc_val))
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Saved DataFrames
# MAGIC Load the pre-saved joined main dataset from phase 3
# MAGIC - We will only load the ***unscaled*** full table becuase we want to undersample before normalizing the numeric columns
# MAGIC - The main data contains all the features and timestamp columns we will use for data splitting and model training/testign

# COMMAND ----------

if dataset == 'prescaled':
    df_joined_all = spark.read.parquet(f"{blob_url}/df_flight_weather_rank{data_suffix}").drop('dep_iata')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add the holiday feature based on a US Holiday Dataset
# MAGIC based on phase 3 feedback, we need to change the holiday indicator by joining a [US Holiday Dataset](https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021?resource=download)
# MAGIC - The US holiday dataset contains US holiday dates from 2004-2021. 
# MAGIC - Since we'd like to mark the day before and after holiday as a holiday in our feature as well, the dataset needs to be tweaked to reflect this
# MAGIC - The US holiday dataset will be joined to our main dataset and the old holiday indicator column will be dropped

# COMMAND ----------

if dataset == 'prescaled':
    #Usually the day before and after holiday are also busy, so we want to indicate +- 1 day of US holiday as holiday
    hd_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/jthsiao@berkeley.edu/US_Holiday_Dates__2004_2021_.csv")
    hd_df = hd_df.filter("Year >= 2015").withColumnRenamed('Holiday','us_holiday').select(['Date','us_holiday'])

    hd_df2 = hd_df.withColumn('day_before',F.col('Date').cast('date')- 1).select('day_before','us_holiday').withColumnRenamed('day_before','Date')
    hd_df3 = hd_df.withColumn('day_after',F.col('Date').cast('date') + 1).select('day_after','us_holiday').withColumnRenamed('day_after','Date')

    hd_df4 = hd_df2.union(hd_df3)

    hd_df = hd_df.union(hd_df4)
    display(hd_df)


# COMMAND ----------

if dataset == 'prescaled':
    df_joined_all = df_joined_all.join(hd_df.select(['Date','us_holiday']), 
                                       df_joined_all.fl_date == hd_df.select(['Date','us_holiday']).Date,'left').drop("Date","holiday")
    display(df_joined_all)

# COMMAND ----------

if dataset == 'prescaled':
    df_joined_all = (df_joined_all.withColumn('is_holiday',F.when(F.col('us_holiday').isNull() == True, F.lit(0)).otherwise(F.lit(1)))
                                .drop('us_holiday')
                   )
    display(df_joined_all)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sampling on Unscaled Dataset
# MAGIC 
# MAGIC #### Undersample
# MAGIC Becasue our dataset is significantly imbalanced (there are many more undelayed flights than delayed flights **22% delayed, 78% not delayed**), and this might potentially affect the modeling result, we decided to undersample to achieve a relatively more balanced dataset for modeling. to do so, we randomly sample a fraction of the non-delay flight entries so that it's about the same number of entries as the delayed flights.
# MAGIC The majority class was randomly downsampled to match 1 to 1 the number of samples in the minority class (delayed flights).  
# MAGIC 
# MAGIC #### SMOTE
# MAGIC Attempted two variations of SMOTE, one leveraging kNN and the other locality sensitive hashing.  The executions to oversample exceeded the capabilities of the cluster.  Reductions in the number elements were made by first applying a random downsampling of the majority class to a ratio of 2 to 1. There is a promising implementation, [Approx-SMOTE](https://github.com/mjuez/approx-smote), which leverages an approximate neighbor search offering 7 to 28x speed improvement, which could be attempted in future work.

# COMMAND ----------

if dataset == 'prescaled' and eda == 'Yes':
    df_joined_all_hist = df_joined_all.select('delay_label').rdd.flatMap(lambda x: x).histogram(2)

# COMMAND ----------

if dataset == 'prescaled' and eda == 'Yes':
    # Loading the Computed Histogram into a Pandas Dataframe for plotting
    plot = pd.DataFrame(
        list(zip(*df_joined_all_hist)), 
        columns=['bin', 'frequency']
    ).set_index(
        'bin'
    ).plot(kind='bar',figsize=(8,8), title='Distribution of Delayed Flights', xlabel='Delay Label', ylabel = 'Frequency');

    for p in plot.patches:
        plot.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at the bar chart above, we can see how the data that we have is obviously imbalanced. We will perform undersampling to address this issue.

# COMMAND ----------

# https://stackoverflow.com/questions/53978683/how-to-undersampling-the-majority-class-using-pyspark
def resample(base_features,ratio,class_field,base_class,stat=False):
    """funciton to undersample with specified ratio"""
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
  
if dataset == 'prescaled':
    df_joined_under = resample(df_joined_all, 1, 'delay_label', 1)
    resample(df_joined_under, 1, 'delay_label', 1, stat=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation among potential features
# MAGIC 
# MAGIC Based on the correlation matrix below for the the potential features we kept, we can see how variables like snow_fall/snow_depth are heavily correlated with daily_precipitation and other weather factors, with a correlation coefficient of about 1.0. This means that including both of these features for our model predictions is not necessary, so we chose to reduce the feature list by removing snowfall information while still retaining the same amount of information. We chose to include the features that had lower correlation coefficients so as not to skew the model predictions and only include the most important features.

# COMMAND ----------

if eda == 'Yes':
    df_pca_train_rep_under = spark.read.parquet(f"{blob_url}/df_pca_train_rep_under{data_suffix}")

# COMMAND ----------

numeric_col = ['hourly_dry_bulb_temperature',
                    'hourly_precipitation',
                    'hourly_relative_humidity',
                    'hourly_visibility',
                    'hourly_wind_speed',
                    'daily_precipitation',
                    'dep_rank']

#these are not included in the features for prediction 'carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay',
#pca_feature_col = ['is_holiday','spring','summer','fall','winter','early_morning','morning','afternoon','evening','night'] + [c+'_scaled' for c in numeric_col]  + ['delay_label']

pca_feature_col =  ['delay_label',
                         'holiday',
                         'spring',
                         'summer',
                         'fall',
                         'winter',
                         'early_morning',
                         'morning',
                         'afternoon',
                         'evening',
                         'night',
                         'hourly_dry_bulb_temperature_scaled',
                         'hourly_precipitation_scaled',
                         'hourly_visibility_scaled',
                         'hourly_wind_speed_scaled',
                         'daily_precipitation_scaled',
                         'daily_snow_depth_scaled',
                         'daily_snow_fall_scaled',
                         'dep_rank_scaled']

if eda == 'Yes':
    # convert to vector column first
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=pca_feature_col, outputCol=vector_col, handleInvalid='skip')
    df_vector = assembler.transform(df_pca_train_rep_under).select(vector_col)

    # get correlation matrix
    matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
    corrmatrix = matrix.toArray().tolist()

    sns.heatmap(corrmatrix, 
            xticklabels=pca_feature_col,
            yticklabels=pca_feature_col).set(title='Weather/Time Correlation Matrix (Pre Feature Selection)')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop the supporting/transition columns, columns that contain information not availalbe 2hrs before scheduled departure time

# COMMAND ----------

if dataset == 'prescaled':
    # select the meaningful independent variables and dependent variable
    # also removed highly correlated variables
    keep_col = ['quarter', 'month', 'day_of_week', 'target_dep_utc_timestamp','airport_iata','dest_iata','is_holiday','spring','summer', 'fall','winter','early_morning','morning','afternoon','evening','night','hourly_dry_bulb_temperature','hourly_precipitation','hourly_relative_humidity','hourly_visibility','hourly_wind_speed','daily_precipitation','dep_rank']
    label = ["delay_label"]
# list out all the numeric columns
numeric_col = ['hourly_dry_bulb_temperature',
                    'hourly_precipitation',
                    'hourly_relative_humidity',
                    'hourly_visibility',
                    'hourly_wind_speed',
                    'daily_precipitation',
                    'dep_rank']

# COMMAND ----------

if dataset == 'prescaled':
    df_under_cols = df_joined_under.select(*keep_col, *label).na.fill(0).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scale/normalize numeric columns on undersampled data
# MAGIC  - Normalization will be applied to the undersample dataset because these are the final data entries for our model training
# MAGIC 
# MAGIC  - We will only normalize numeric columns
# MAGIC  - In order for the normalization to work, the dataset cannot contain NaNs. so one step before normalization is to fill in all Nulls and NaNs with 0s
# MAGIC  - The normalization technique we choose is the Z-score standardization, using the following formula for each column:
# MAGIC    $${\(\frac {X_i - \bar{X}} s\)}$$

# COMMAND ----------

if dataset == 'prescaled':
    # fill out the NA with 0
    #df_feature_scaled = df_under_cols.fillna(0,subset = numeric_col)

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

        df_under_cols = (pipeline.fit(df_under_cols).transform(df_under_cols)
           .withColumn(c+'_scaled',unlist(c+'_Scaled')).drop(c+'_Vect',c))
    
    df_under_feature_scaled = df_under_cols
#write to storage
#df_under_feature_scaled.write.mode('overwrite').parquet(f"{blob_url}/df_under_feature_scaled{data_suffix}")
    

# COMMAND ----------

if dataset == 'prescaled':
    display(df_under_feature_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read the ***scaled*** dataset (prebuilt)

# COMMAND ----------

if dataset == 'scaled':
    df_under_feature_scaled =spark.read.parquet(f"{blob_url}/df_under_feature_scaled{data_suffix}")

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import col,isnan,when,count
#c not in {'column_1', 'column_2', 'column_3'}
# df2 = df.select([count(when(col(c).contains('None') | \
#                             col(c).contains('NULL') | \
#                             (col(c) == '' ) | \
#                             col(c).isNull() | \
#                             isnan(c), c 
#                            )).alias(c)
#                     for c in df.columns])
# df2.show()
first_col = {'actual_dep_utc_timestamp','target_dep_utc_timestamp','wheels_off_utc_timestamp','wheels_on_utc_timestamp','target_arr_utc_timestamp','actual_arr_utc_timestamp','weather_utc_timestamp'}
#[cols for cols in df.columns if cols not in first_col]
df_joined_all.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_joined_all.columns if c not in first_col]).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation After Removing Dependent Features
# MAGIC After removing snowfall information (which is included in precipitation) and gust wind (which is highly correlated wtih wind), we don't see obvious multicollinearity in our features

# COMMAND ----------

if eda == 'Yes':

    numeric_col = ['hourly_dry_bulb_temperature',
                        'hourly_precipitation',
                        'hourly_relative_humidity',
                        'hourly_visibility',
                        'hourly_wind_speed',
                        'daily_precipitation',
                        'dep_rank']

    #these are not included in the features for prediction 'carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay',
    #pca_feature_col = ['is_holiday','spring','summer','fall','winter','early_morning','morning','afternoon','evening','night'] + [c+'_scaled' for c in numeric_col]  + ['delay_label']

    pca_feature_col =  ['delay_label',
                             'is_holiday',
                             'spring',
                             'summer',
                             'fall',
                             'winter',
                             'early_morning',
                             'morning',
                             'afternoon',
                             'evening',
                             'night',
                             'hourly_dry_bulb_temperature_scaled',
                             'hourly_precipitation_scaled',
                             'hourly_visibility_scaled',
                             'hourly_wind_speed_scaled',
                             'daily_precipitation_scaled',
                             'dep_rank_scaled']

    # convert to vector column first
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=pca_feature_col, outputCol=vector_col, handleInvalid='skip')
    df_vector = assembler.transform(df_under_feature_scaled).select(vector_col)

    # get correlation matrix
    matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
    corrmatrix = matrix.toArray().tolist()

    sns.heatmap(corrmatrix, 
            xticklabels=pca_feature_col,
            yticklabels=pca_feature_col).set(title='Weather/Time Correlation Matrix (Post Feature Selection)')

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA on Finalized Numeric Features
# MAGIC Histogram of the final numeric features show that datapoints are centered around 0, standard scaling was done correctly

# COMMAND ----------

import seaborn as sns
import pandas as pd
holiday_df = (df_under_feature_scaled.sample(False,0.001,81)).toPandas()

# COMMAND ----------

holiday_df[holiday_df.delay_label == 1]

# COMMAND ----------

import matplotlib.pyplot as plt
fig, ax =plt.subplots(1,4, figsize = (20,4))
for c,i in zip(['early_morning','morning','afternoon','evening'],range(4)):
    sns.countplot(x = 'delay_label', data = holiday_df[holiday_df[c] == 1],ax = ax[i])
    ax[i].set_title(c)

# COMMAND ----------

import matplotlib.pyplot as plt
fig, ax =plt.subplots(1,3, figsize = (20,5))
for c,i in zip(['hourly_visibility_scaled','hourly_wind_speed_scaled', 'daily_precipitation_scaled'],range(3)):
    sns.boxplot(x = 'delay_label', y = c,data = holiday_df, ax = ax[i])
    ax[i].set_title(c)

# COMMAND ----------

numeric_col = ['hourly_dry_bulb_temperature',
                    'hourly_precipitation',
                    'hourly_relative_humidity',
                    'hourly_visibility',
                    'hourly_wind_speed',
                    'daily_precipitation',
                    'dep_rank']
    
scaled_col = [c+'_scaled' for c in numeric_col]
sns.countplot(x = 'day_part',data=holiday_df[holiday_df.delay_label == 1],palette='rainbow')

# COMMAND ----------

holiday_df.plot(kind = 'bar')

# COMMAND ----------

if eda == 'Yes':
    numeric_col = ['hourly_dry_bulb_temperature',
                    'hourly_precipitation',
                    'hourly_relative_humidity',
                    'hourly_visibility',
                    'hourly_wind_speed',
                    'daily_precipitation',
                    'dep_rank']
    
    scaled_col = [c+'_scaled' for c in numeric_col]
    feature_pd = (df_under_feature_scaled.select(*scaled_col).sample(False,0.01,81)).toPandas()
    import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(2,4, figsize=(20,10))
    fig=plt.figure(figsize=(20,10))
    for i, c in zip(range(4),scaled_col[:4]):
        ax=fig.add_subplot(2,5,i+1)
        feature_pd[c].hist(bins=10,ax=ax)
        ax.set_title(c)
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()
    fig=plt.figure(figsize=(20,10))
    for i, c in zip(range(5),scaled_col[4:]):
        ax=fig.add_subplot(2,5,i+1)
        feature_pd[c].hist(bins=10,ax=ax)
        ax.set_title(c)
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary of airport rank
# MAGIC ORD and ATL are highst ranked airports. Which are consistent with our own experiements and news reporting that these are the busiest airports in the US

# COMMAND ----------

if dataset == 'prescaled' and eda == 'Yes':
    year = udf(lambda d: d.year, T.IntegerType())

    df_popularity = df_joined_all.withColumn('year', year(df_joined_all.target_dep_utc_timestamp))

    airport_popularity = df_popularity.select(['dep_rank', 'airport_iata', 'year'])\
                           .drop_duplicates()\
                           .orderBy(col("dep_rank").desc())

    display(airport_popularity.select(['dep_rank', 'airport_iata']).drop_duplicates().orderBy(col("dep_rank").desc()).head(10))

# COMMAND ----------

if dataset == 'prescaled' and eda == 'Yes':
    airport_popularity = df_joined_all.select(['dep_rank', 'airport_iata', 'delay_label'])\
        .groupBy('airport_iata', 'dep_rank')\
        .agg(F.sum('delay_label'))\
        .orderBy(col('dep_rank').desc())\
        .select(['dep_rank', 'airport_iata', 
              "sum(delay_label)", 
              F.monotonically_increasing_id().alias('rank')])
    
    display(airport_popularity.orderBy(col("sum(delay_label)").desc()).head(10))                  

# COMMAND ----------

if dataset == 'prescaled' and eda == 'Yes':
    airport_popularity = df_joined_all.select(['dep_rank', 'airport_iata', 'delay_label'])\
                                      .groupBy('airport_iata', 'dep_rank')\
                                      .agg(F.sum('delay_label'))\
                                      .orderBy(col('dep_rank').desc())\
                                      .select(['dep_rank', 'airport_iata', "sum(delay_label)", F.monotonically_increasing_id().alias('rank')])

    geo = (airport_popularity.orderBy(col("sum(delay_label)").desc()).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Split (Train/Test Sets)
# MAGIC Split by year becuase we are working with a time series

# COMMAND ----------

year = udf(lambda d: d.year, T.IntegerType())
year_order = udf(lambda d: d.year%100-15, T.IntegerType())
year_quarter = udf(lambda d,q: d.year*10+q, T.IntegerType())
df_final_scaled = df_under_feature_scaled.withColumn('year', year(df_under_feature_scaled.target_dep_utc_timestamp))\
                     .withColumn('year_order', year_order(df_under_feature_scaled.target_dep_utc_timestamp))\
                     .withColumn('year_quarter', year_quarter(df_under_feature_scaled.target_dep_utc_timestamp, df_under_feature_scaled.quarter))

df_test = df_final_scaled.filter(col('year') == 2021)
df_train= df_final_scaled.filter(col('year') < 2021)

# COMMAND ----------

display(df_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation for Logistic Regression and Random Forest

# COMMAND ----------

if model_selection in ('LogisticRegression', 'RandomForrest', 'All'):
    df_under_feature_scaled = spark.read.parquet(f"{blob_url}/df_under_feature_scaled")
    year = udf(lambda d: d.year, T.IntegerType())
    year_order = udf(lambda d: d.year%100-15, T.IntegerType())
    year_quarter = udf(lambda d,q: d.year*10+q, T.IntegerType())
    df_final_scaled = df_under_feature_scaled.withColumn('year', year(df_under_feature_scaled.target_dep_utc_timestamp))\
                         .withColumn('year_order', year_order(df_under_feature_scaled.target_dep_utc_timestamp))\
                         .withColumn('year_quarter', year_quarter(df_under_feature_scaled.target_dep_utc_timestamp, df_under_feature_scaled.quarter))

    df_final_test = df_final_scaled.filter(col('year') == 2021)
    df_final_train = df_final_scaled.filter(col('year') < 2021)
    df_final_test_sample = df_final_test.sample(False, 0.1)
    df_final_train_sample = df_final_train.sample(False, 0.1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic Logistic Regression (PCA on undersample data)

# COMMAND ----------

if model_selection in ('LogisticRegression', 'RandomForrest', 'All'):
    numeric_col = ['hourly_dry_bulb_temperature', 'hourly_precipitation', 'hourly_relative_humidity', 'hourly_visibility', 'hourly_wind_speed', 'daily_precipitation', 'dep_rank']
    feature_col = ['is_holiday','spring','summer', 'fall','winter','early_morning','morning','afternoon','evening','night'] + [c+'_scaled' for c in numeric_col]

    pca_assembler = VectorAssembler(inputCols=feature_col, outputCol="features", handleInvalid = "skip")
    pca_model = PCA(k = 5,inputCol = "features", outputCol = "pca_features")

    pca_lr = LogisticRegression(maxIter=10, regParam=0.3).setLabelCol('delay_label')
    pca_pipeline = Pipeline(stages=[pca_assembler,pca_model, pca_lr])
    pca_reg = pca_pipeline.fit(df_train)
    pca_pred = pca_reg.transform(df_test)

# COMMAND ----------

if model_selection in ('LogisticRegression', 'RandomForrest', 'All'):
    recall = MulticlassClassificationEvaluator(metricName="weightedRecall", labelCol="delay_label")
    precision = MulticlassClassificationEvaluator(metricName="weightedPrecision", labelCol="delay_label")
    f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="delay_label")

    print(f"Weighted Recall: {recall.evaluate(pca_pred)}")
    print(f"Weighted Precision: {precision.evaluate(pca_pred)}")
    print(f"F1: {f1.evaluate(pca_pred)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Custom CV Data Preparation on Time Series with Rolling Basis

# COMMAND ----------

if model_selection in ('LogisticRegression', 'RandomForrest', 'All'):
    # create dictionary of dataframes for custom cv fn to loop through
    # assign train and test based on time series split 

    d = {}

    d['df1'] = df_final_train_sample.filter(df_final_train_sample.year <= 2016)\
                       .withColumn('cv', F.when(df_final_train_sample.year == 2015, 'train')
                                             .otherwise('test'))

    d['df2'] = df_final_train_sample.filter(df_final_train_sample.year <= 2017)\
                       .withColumn('cv', F.when(df_final_train_sample.year <= 2016, 'train')
                                             .otherwise('test'))

    d['df3'] = df_final_train_sample.filter(df_final_train_sample.year <= 2018)\
                       .withColumn('cv', F.when(df_final_train_sample.year <= 2017, 'train')
                                             .otherwise('test'))

    d['df4'] = df_final_train_sample.filter(df_final_train_sample.year <= 2019)\
                       .withColumn('cv', F.when(df_final_train_sample.year <= 2018, 'train')
                                             .otherwise('test'))

    d['df5'] = df_final_train_sample.filter(df_final_train_sample.year <= 2020)\
                       .withColumn('cv', F.when(df_final_train_sample.year <= 2019, 'train')
                                             .otherwise('test'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Feature Types

# COMMAND ----------

if model_selection in ('LogisticRegression', 'RandomForrest', 'All'):
    categoricals = ['quarter', 'month', 'day_of_week', 'airport_iata', 'dest_iata', 'is_holiday','spring','summer', 'fall','winter','early_morning','morning','afternoon','evening', 'night', 'year', 'year_order', 'year_quarter']
    numerics =['hourly_dry_bulb_temperature_scaled','hourly_precipitation_scaled','hourly_relative_humidity_scaled','hourly_visibility_scaled','hourly_wind_speed_scaled','daily_precipitation_scaled', 'dep_rank_scaled']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logitstic Regression Model (CV + Hyper Tune)

# COMMAND ----------

if model_selection in ('LogisticRegression', 'All'):
    evaluator = BinaryClassificationEvaluator()

    ## Current possible ways to handle categoricals in string indexer is 'error', 'keep', and 'skip'
    indexers = map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), categoricals)
    ohes = map(lambda c: OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class"),categoricals)
    imputers = Imputer(inputCols = numerics, outputCols = numerics)

    # Establish features columns
    featureCols = list(map(lambda c: c+"_class", categoricals)) + numerics

    # Build the stage for the ML pipeline
    model_matrix_stages = list(indexers) + list(ohes) + [imputers] + \
                         [VectorAssembler(inputCols=featureCols, outputCol="features"), StringIndexer(inputCol="delay_label", outputCol="label")]

    # pca model
    pca_model = PCA(k = 5,inputCol = "features", outputCol = "pca_features")

    # Use logistic regression 
    lr = LogisticRegression(featuresCol = "pca_features")

    # Build our ML pipeline
    pipeline = Pipeline(stages=model_matrix_stages + [pca_model, lr])

    # Build the parameter grid for model tuning
    grid = ParamGridBuilder() \
                  .addGrid(lr.regParam, [2.0, 1.0, 0.1, 0.01]) \
                  .addGrid(lr.maxIter, [1, 5, 10]) \
                  .addGrid(lr.elasticNetParam, [0, 0.5, 0.8, 1]) \
                  .build()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Custom CV and Hyper Tuning

# COMMAND ----------

if model_selection in ('LogisticRegression', 'All'):
    cv = CustomCrossValidator(estimator=pipeline, 
                              estimatorParamMaps=grid, 
                              evaluator=evaluator,
                              splitWord = ('train', 'test'), 
                              cvCol = 'cv', 
                              parallelism=4)

    cvModel = cv.fit(d)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select the Best Model Parameters and Train the Full data set

# COMMAND ----------

if model_selection in ('LogisticRegression', 'All'):
    # initialize the random forest for pipeline
    lr = LogisticRegression(labelCol="label",
                            featuresCol="pca_features",
                            regParam=0.01,
                            maxIter=1,
                            elasticNetParam=1.0)
    pipeline = Pipeline(stages=model_matrix_stages + [pca_model, lr])
    lr_model = pipeline.fit(df_final_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict and Evaluate

# COMMAND ----------

if model_selection in ('LogisticRegression', 'All'):
    prediction = lr_model.transform(df_final_test)

# COMMAND ----------

if model_selection in ('LogisticRegression', 'All'):
    recall = MulticlassClassificationEvaluator(metricName="weightedRecall", labelCol="delay_label").evaluate(prediction)
    precision = MulticlassClassificationEvaluator(metricName="weightedPrecision", labelCol="delay_label").evaluate(prediction)
    f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="delay_label").evaluate(prediction)
    auc = BinaryClassificationEvaluator().evaluate(prediction)

    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1: {f1}")
    print(f"AUC: {auc}")

# COMMAND ----------

if model_selection in ('LogisticRegression', 'All'):
    plot_roc_curve(prediction)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Model (CV + Hyper Tune)

# COMMAND ----------

if model_selection in ('RandomForrest', 'All'):
    evaluator = BinaryClassificationEvaluator()

    ## Current possible ways to handle categoricals in string indexer is 'error', 'keep', and 'skip'
    indexers = map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), categoricals)
    ohes = map(lambda c: OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class"),categoricals)
    imputers = Imputer(inputCols = numerics, outputCols = numerics)

    # Establish features columns
    featureCols = list(map(lambda c: c+"_class", categoricals)) + numerics

    # Build the stage for the ML pipeline
    model_matrix_stages = list(indexers) + list(ohes) + [imputers] + \
                         [VectorAssembler(inputCols=featureCols, outputCol="features"), StringIndexer(inputCol="delay_label", outputCol="label")]

    # pca model
    pca_model = PCA(k = 5,inputCol = "features", outputCol = "pca_features") 

    # initialize the random forest for pipeline
    rf = RandomForestClassifier(labelCol="label", featuresCol="pca_features")
    pipeline = Pipeline(stages=model_matrix_stages + [pca_model, rf])

    # Build the parameter grid for model tuning
    grid = ParamGridBuilder() \
                  .addGrid(rf.bootstrap, [True, False]) \
                  .addGrid(rf.maxDepth, [5, 20]) \
                  .addGrid(rf.featureSubsetStrategy, ['auto', 'sqrt']) \
                  .addGrid(rf.numTrees, [20, 40]) \
                  .build()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Custom CV and Hyper Tuning

# COMMAND ----------

if model_selection in ('RandomForrest', 'All'):
    cv_rf = CustomCrossValidator(estimator=pipeline, 
                              estimatorParamMaps=grid, 
                              evaluator=evaluator,
                              splitWord = ('train', 'test'), 
                              cvCol = 'cv', 
                              parallelism=4)

    cvRfModel = cv_rf.fit(d)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select the Best Model Parameters and Train the Full data set

# COMMAND ----------

if model_selection in ('RandomForrest', 'All'):
    # initialize the random forest for pipeline
    rf = RandomForestClassifier(labelCol="label", 
                                featuresCol="pca_features", 
                                numTrees=40, 
                                bootstrap=True, 
                                maxDepth=5, 
                                featureSubsetStrategy='sqrt')
    pipeline = Pipeline(stages=model_matrix_stages + [pca_model, rf])
    rf_model = pipeline.fit(df_final_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict and Evaluate

# COMMAND ----------

if model_selection in ('RandomForrest', 'All'):
    prediction = rf_model.transform(df_final_test)
