# Databricks notebook source
# MAGIC %md
# MAGIC #Phase 2 - Summary
# MAGIC 
# MAGIC In phase 2, the group worked on visualizing more features of all the tables, along with performing the main join across all the tables. We first began by joining the airport and weather station data, and then joined the weather data to the airport/weather station data, and then adding the flight data to the combination of these previously joined data sets. For feature engineering, we standardized the format of timestamps, grouped weather data into hourly stats, converted categorical variables, normalized numerical columns, and imputed null values with 0. We also implemented PCA for dimensionality reduction and plan to combine the resulting principle components with the delay_labels to use for modeling. Finally, we created our baseline logistic regression model by first splitting our data into training/test set, performing feature selection, assembling the features as an input vector to feed into our LM (ridge regression model), and then build the pipeline. The F1_scores of our model are 0.74014 (training set) and 0.74095 (testing set), which are satisfactory as they are generated based on a draft logistic regression model without hyper-tuning and cross-validation on Time Series.
# MAGIC 
# MAGIC phase 1 notebook https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/3326810404140493/command/28620905305976

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

# MAGIC %md
# MAGIC ## Additional EDA pre join

# COMMAND ----------

# MAGIC %md
# MAGIC ### Flights Dataset

# COMMAND ----------

print("Departure Delays vs. Day of the Week")
display(df_flights)

# COMMAND ----------

# MAGIC %md
# MAGIC There seems to be some trend in flight delays among the entire dataset of flight information. For the interest of the classification outcome (delay or not), the average number of delays seem to be highest on Monday and Sunday, and less frequent number of delays on Tuesday, Wednesday, and Fridays.

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the graph above, the summer months seem to have the highest average number of departure delays (June, July, and August), in addition to December also having a relatively high number of average delays. This may be due to more people traveling during December for holidays, and also during the summer, which could cause an increase in the amount of flights and thereby increasing the amount of delays in proportion to the other months. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather dataset

# COMMAND ----------

cnt_distinct = df_weather.select('Name').distinct().count()

# COMMAND ----------

print("Box Plot Daily Average Wind Speed")
display(df_weather)

# COMMAND ----------

print("Hourly Precipitation Distribution")
display(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table Joining
# MAGIC 
# MAGIC To join the weather and flight datasets together, additional information was required. In their raw state there was no common key to link the two datasets together. The weather dataset contained a “station id”, which is the station identifier of the station that collected the measurements. The flight dataset contained the “origin” which is the IATA code of the departing flight. To link the two datasets, a weather station and airport dataset was introduced to bridge the gap i.e. Weather <-> Weather Station <-> Airport <-> Flights.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Airport Code + Weather Station
# MAGIC 
# MAGIC After reviewing the weather station dataset, which contained station information and neighboring station information.  Since the station information was missing the ICAO code, the neighboring station information was utilized as it contained the ICAO code and station identifier.  By leveraging the [Airport Database](https://www.partow.net/miscellaneous/airportdatabase/index.html#Downloads), containing ICAO and IATA codes, the two datasets were joined via ICAO code as the key to provide a new table containing ICAO codes, IATA codes, and station identifiers. (Partow, 2017)  To help reduce the dataset size duplicate rows were dropped.

# COMMAND ----------

# Removed unused columns, standardize column names, drop duplicates

used_station_columns = ['neighbor_id', 'neighbor_name', 'neighbor_state', 
                        'neighbor_call', 'neighbor_lat', 'neighbor_lon' ]

df_stations_p = df_stations.select(*used_station_columns)\
                           .withColumnRenamed('neighbor_id', 'station_id')\
                           .withColumnRenamed('neighbor_name', 'station_name')\
                           .withColumnRenamed('neighbor_state', 'station_state')\
                           .withColumnRenamed('neighbor_call', 'station_call')\
                           .withColumnRenamed('neighbor_lat', 'station_lat')\
                           .withColumnRenamed('neighbor_lon', 'station_lon')\
                           .drop_duplicates()

df_airports_s = df_airports.select('call', 'iata')
df_stations_p = df_stations_p.join(df_airports_s, df_stations_p.station_call == df_airports_s.call, 'inner')

used_station_columns = ['station_id', 'station_name', 'station_state', 
                        'station_call', 'station_lat', 'station_lon', 
                        'iata' ]

df_stations_p = df_stations_p.select(*used_station_columns)\
                             .withColumnRenamed('iata', 'airport_iata')
   
print(df_stations_p.count())
display(df_stations_p)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather + (Airport + Weather Station)
# MAGIC 
# MAGIC After obtaining the Airport + Weather Station dataset, the IATA codes were added to the weather dataset via a join operation on station identifier.  The dataset size was reduced by first removing the columns identified for pruning from EDA activities; then duplicate rows were dropped.  Nulls in specific columns were converted to zeros in cases identified to make sense such as hourly_wind_gust_speed which would be set to null when the current wind speed was not classified as a gust.   Additional time related columns were added, utc timestamp and a time adjusted key rounded to the nearest hour in the format YYYYMMDDHH. Multiple rows per station within the same hour were aggregated based on various techniques such as selecting the max value for precipitation or the mean value for humidity.

# COMMAND ----------

df_station_weather = df_stations_p.join(df_weather, df_stations_p.station_id == df_weather.STATION, "inner")
display(df_station_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weather Raw Feature Extraction
# MAGIC After review of the weather dataset, several columns were identified early for removal such as weekly and monthly metrics which are believe unapplicable to the project goals of using immediate weather information to predict flight delays.  In the cell below, the weather data was limited to the following features.  

# COMMAND ----------

# Removed unused columns, standardize column names, drop duplicates

used_station_weather_columns = ['station_id', 'station_name', 'station_state', 
                                'station_call', 'station_lat', 'station_lon',
                                'airport_iata',
                                'DATE',
                                'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyPresentWeatherType',
                                'HourlyRelativeHumidity', 'HourlySkyConditions', 'HourlyVisibility',
                                'HourlyWindGustSpeed', 'HourlyWindSpeed',
                                'DailyPrecipitation', 'DailySnowDepth', 'DailySnowfall']

df_station_weather_p = df_station_weather.select(*used_station_weather_columns)\
                          .withColumnRenamed('DATE', 'local_datetime')\
                          .withColumnRenamed('HourlyDryBulbTemperature', 'hourly_dry_bulb_temperature')\
                          .withColumnRenamed('HourlyPrecipitation', 'hourly_precipitation')\
                          .withColumnRenamed('HourlyPresentWeatherType', 'hourly_present_weather_type')\
                          .withColumnRenamed('HourlyRelativeHumidity', 'hourly_relative_humidity')\
                          .withColumnRenamed('HourlySkyConditions', 'hourly_sky_conditions')\
                          .withColumnRenamed('HourlyVisibility', 'hourly_visibility')\
                          .withColumnRenamed('HourlyWindGustSpeed', 'hourly_wind_gust_speed')\
                          .withColumnRenamed('HourlyWindSpeed', 'hourly_wind_speed')\
                          .withColumnRenamed('DailyPrecipitation', 'daily_precipitation')\
                          .withColumnRenamed('DailySnowDepth', 'daily_snow_depth')\
                          .withColumnRenamed('DailySnowfall', 'daily_snow_fall')\
                          .drop_duplicates()

print(df_station_weather_p.count())

display(df_station_weather_p)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Missing Data Management
# MAGIC 
# MAGIC The weather dataset and many fields populated with nulls.  A majority of the columns of weather data were identified as not useful such as yearly and monthly metrics, backup related columns, etc.
# MAGIC * 'hourly_wind_gust_speed': Spikes in wind speeds above 9 knots lasting less than 20 are classified as gusts, else the column is null. Converted nulls to 0. 
# MAGIC * 'daily_precipitation', 'daily_snow_depth', 'daily_snow_fall': Daily metrics which appear contain many null values.  This could be because the data is not reported.  Converted the nulls 0 which would be equivalent to these features not affecting a flight.
# MAGIC 
# MAGIC The following cell casted the numeric columns and filled nulls with zeros where approriate such as gust speed when winds were not classified as gusty or values were not provided.

# COMMAND ----------

df_station_weather_p = df_station_weather_p.withColumns({
                                  'hourly_dry_bulb_temperature': df_station_weather_p.hourly_dry_bulb_temperature.cast(T.IntegerType()),
                                  'hourly_precipitation': df_station_weather_p.hourly_precipitation.cast(T.DoubleType()),
                                  'hourly_relative_humidity': df_station_weather_p.hourly_relative_humidity.cast(T.IntegerType()),                     
                                  'hourly_visibility': df_station_weather_p.hourly_visibility.cast(T.DoubleType()),
                                  'hourly_wind_gust_speed': df_station_weather_p.hourly_wind_gust_speed.cast(T.IntegerType()),
                                  'hourly_wind_speed': df_station_weather_p.hourly_wind_speed.cast(T.IntegerType()),
                                  'daily_precipitation': df_station_weather_p.hourly_wind_gust_speed.cast(T.DoubleType()),
                                  'daily_snow_depth': df_station_weather_p.hourly_wind_gust_speed.cast(T.IntegerType()),
                                  'daily_snow_fall': df_station_weather_p.hourly_wind_gust_speed.cast(T.DoubleType())})
df_station_weather_p = df_station_weather_p.na.fill(value=0, subset=['hourly_wind_gust_speed', 'daily_precipitation', 'daily_snow_depth', 'daily_snow_fall'])
df_station_weather_p.count()

# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC ### UTC Timestamp Conversion
# MAGIC The following cell converts the weather aquistion time to utc as a timestamp.  A special time stamp key 'weather_ymdh' which is a time equivalent string with format YYYYMMDDHH.  The 'weather_ymdh' time is rounded to the nearest hour.

# COMMAND ----------

# Convert times to utc

@udf(returnType=T.TimestampType())
def timestamp_to_utc(iata, local_datetime):
  dt = datetime.datetime.fromisoformat(local_datetime)
  try:
    apt = airporttime.AirportTime(iata_code=iata)
  except:
    return datetime.datetime.utcfromtimestamp(0)
  tz_aware_utc_time = apt.to_utc(dt)
  return tz_aware_utc_time

df_station_weather_p = df_station_weather_p.withColumn('weather_utc_timestamp', timestamp_to_utc(df_station_weather_p.airport_iata, df_station_weather_p.local_datetime))
df_station_weather_p = df_station_weather_p.filter(df_station_weather_p.weather_utc_timestamp != datetime.datetime.utcfromtimestamp(0))

timedelta_half_hour = datetime.timedelta(minutes=30)
@udf(returnType=T.StringType())
def ymdh_round(dt, adjust=timedelta_half_hour):
  d = dt + adjust
  return f'{d.year}{d.month:02d}{d.day:02d}{d.hour:02d}'
df_station_weather_p = df_station_weather_p.withColumn('weather_ymdh', ymdh_round(df_station_weather_p.weather_utc_timestamp))

print(df_station_weather_p.count())
display(df_station_weather_p)

# COMMAND ----------

df_station_weather_p.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Aggregation
# MAGIC 
# MAGIC After having limited the number of weather features and performing time conversions and creation of special time based key, the following cell aggregates the weather information per hour at each station.  The aggregation depends on the type of information such as for temperature the mean is taken where as wind speed the max value as a high wind value could trigger weather condition delay of a flight.  Daily metrics were aggregated as max although in some cases these values would not change much within an hour and are used to identify situations where the airport is impacted due to poor weather conditions throughout the day.

# COMMAND ----------

weather_agg = {c:'first' for c in df_station_weather_p.columns}
weather_agg.update({'hourly_dry_bulb_temperature': 'mean',
                    'hourly_precipitation': 'max',
                    'hourly_relative_humidity': 'mean',
                    'hourly_visibility': 'max',
                    'hourly_wind_gust_speed': 'max',
                    'hourly_wind_speed': 'max',
                    'daily_precipitation': 'max',
                    'daily_snow_depth': 'max',
                    'daily_snow_fall': 'max'})

df_station_weather_p = df_station_weather_p.groupBy('weather_ymdh', 'airport_iata') \
                                           .agg(weather_agg)
display(df_station_weather_p)

# COMMAND ----------

df_station_weather_p = df_station_weather_p.drop('weather_ymdh', 'airport_iata')
station_weather_rename_columns = {c: re.findall(r'\((.*)\)', c)[0] for c in df_station_weather_p.columns}

print(station_weather_rename_columns)
for k, n in station_weather_rename_columns.items():
  df_station_weather_p = df_station_weather_p.withColumnRenamed(k, n)
display(df_station_weather_p)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Save/Load

# COMMAND ----------

# Uncomment to rewrite the parquet
# Uncomment to rewrite the parquet
# df_station_weather_p.write.mode('overwrite').parquet(f"{blob_url}/df_station_weather_p{data_suffix}")
df_station_weather_p = spark.read.parquet(f"{blob_url}/df_station_weather_p{data_suffix}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing Data Management
# MAGIC 
# MAGIC ### Flights
# MAGIC 
# MAGIC The majority of flight data was well populated. There were flights that had departure times and arrive times populated with nulls.  Upon further investigation, these were because the flights were cancelled.  In this case, cancelled flights were imputed as a factor to feature 'delay_label' for modeling. The columns that were null because of cancellations were not used as features.

# COMMAND ----------

df_flight_weather_p = spark.read.parquet(f"{blob_url}/df_flight_weather_p")
df_station_weather_p = spark.read.parquet(f"{blob_url}/df_station_weather_p")
df_flights_p = spark.read.parquet(f"{blob_url}/df_flights_p")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC 
# MAGIC ### Flights
# MAGIC - Standardize format for timestamp to prepare for join and time slicing
# MAGIC - Add early morning, morning, afternoon, evening and late night based on scheduled departure time
# MAGIC - Identify holidays
# MAGIC - Identify flight season
# MAGIC - Identify extra busy airports
# MAGIC 
# MAGIC ### Weather
# MAGIC - Narrow down to hourly weather for each airport
# MAGIC 
# MAGIC ### Joined flights+weather
# MAGIC - normalize each of the numeric columns so that when we run algorithms using gradient descent, we won't run into non-convergence. The formula for normalizing is:
# MAGIC $${\(\frac {X_i - \bar{X}} s\)}$$
# MAGIC 
# MAGIC Note: Dep_delay rows are not removed, because these are cancelled flights. Cancelled flights are labeled as "1"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Flight Raw Features Extraction & hot encoding
# MAGIC The following cell extracted columns of interest provided when doing EDA. The times were converted to utc. A special time based key was created which is two hours prior to the scheduled departure time saved as a string in the format YYYYMMDDHH to later be used for adding weather data to each flight. Additional features were added such as season, parts of the day, and holidays.

# COMMAND ----------

# Convert times to utc
adjust_max_time_udf = udf(lambda t: 2359 if t == 2400 else t, T.IntegerType())
# adjust_max_time = lambda t: 2359 if t == 2400 else t

df_flights_t = df_flights.withColumns({'CRS_DEP_TIME': adjust_max_time_udf(df_flights.CRS_DEP_TIME),
                                       'DEP_TIME': adjust_max_time_udf(df_flights.DEP_TIME),
                                       'WHEELS_OFF': adjust_max_time_udf(df_flights.WHEELS_OFF),
                                       'WHEELS_ON': adjust_max_time_udf(df_flights.WHEELS_ON),
                                       'CRS_ARR_TIME': adjust_max_time_udf(df_flights.CRS_ARR_TIME),
                                       'ARR_TIME': adjust_max_time_udf(df_flights.ARR_TIME)})

@udf(returnType=T.TimestampType())
def flight_timestamp_to_utc(iata, day, time):
  if not time:
    return None
  dt = datetime.datetime.fromisoformat(f'{day[:10]} {int(time / 100):02d}:{int(time % 100):02d}:00')
  try:
    apt = airporttime.AirportTime(iata_code=iata)
  except:
    return None
  tz_aware_utc_time = apt.to_utc(dt)
  return tz_aware_utc_time

df_flights_t = df_flights_t.na.drop(subset=['FL_DATE', 'CRS_DEP_TIME']) \
                           .withColumns({'target_dep_utc_timestamp': flight_timestamp_to_utc(df_flights_t.ORIGIN, df_flights_t.FL_DATE, df_flights_t.CRS_DEP_TIME),
                                         'actual_dep_utc_timestamp': flight_timestamp_to_utc(df_flights_t.ORIGIN, df_flights_t.FL_DATE, df_flights_t.DEP_TIME),
                                         'wheels_off_utc_timestamp': flight_timestamp_to_utc(df_flights_t.ORIGIN, df_flights_t.FL_DATE, df_flights_t.WHEELS_OFF),
                                         'wheels_on_utc_timestamp': flight_timestamp_to_utc(df_flights_t.ORIGIN, df_flights_t.FL_DATE, df_flights_t.WHEELS_ON),
                                         'target_arr_utc_timestamp': flight_timestamp_to_utc(df_flights_t.ORIGIN, df_flights_t.FL_DATE, df_flights_t.CRS_ARR_TIME),
                                         'actual_arr_utc_timestamp': flight_timestamp_to_utc(df_flights_t.ORIGIN, df_flights_t.FL_DATE, df_flights_t.ARR_TIME)
                                        })
#                           .na.drop(subset=['target_dep_utc_timestamp']) \
display(df_flights_t)
print(df_flights_t.count())

# COMMAND ----------

#help functions for feature engineering
holidays = ['1-1','12-31','1-2','7-3','7-4','7-5','11-27','11-28','11-29','12-24','12-25','12-26']
def extract_hour(t):
  return t//100

extract_hour_udf = udf(extract_hour, T.IntegerType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add categoriocal variables to Flights Data
# MAGIC Add season, holiday, day park categorical variables and created dummy variables for those

# COMMAND ----------

# Removed unused columns, standardize column names, drop duplicates

used_flight_columns = ['QUARTER','MONTH','DAY_OF_WEEK','FL_DATE',
                       'OP_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN', 'DEST',
                       'target_dep_utc_timestamp', 'actual_dep_utc_timestamp',
                       'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY',
                       'WHEELS_OFF', 'wheels_off_utc_timestamp',
                       'wheels_on_utc_timestamp', 'WHEELS_ON', 
                       'target_arr_utc_timestamp', 'actual_arr_utc_timestamp',
                       'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY',
                       'CANCELLED', 'DIVERTED','CANCELLATION_CODE',
                       'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 
                       'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
                      ]
#New features added
#delay_label: 1 for departure delay >1, 0 else
#holiday: 1 if it falls into +-1 day of New Year's Day, 4th of July, Thanksgiving, and Christmas
#season: dummy variables for all seasons, spring, summer, fall and winter
#day_part: dummy variables for depart time in early morning, morning, afternoon, evening and night

df_flights_p = df_flights_t.select(*used_flight_columns)\
                          .withColumnRenamed('QUARTER','quarter')\
                          .withColumnRenamed('MONTH','month')\
                          .withColumnRenamed('DAY_OF_WEEK','day_of_week')\
                          .withColumnRenamed('FL_DATE','fl_date')\
                          .withColumnRenamed('OP_CARRIER', 'airline')\
                          .withColumnRenamed('TAIL_NUM', 'tail_num')\
                          .withColumnRenamed('OP_CARRIER_FL_NUM', 'flight_num')\
                          .withColumnRenamed('ORIGIN', 'airport_iata')\
                          .withColumnRenamed('DEST', 'dest_iata')\
                          .withColumnRenamed('CRS_DEP_TIME', 'target_dep_time')\
                          .withColumnRenamed('DEP_TIME', 'actual_dep_time')\
                          .withColumnRenamed('DEP_DELAY', 'dep_delay')\
                          .withColumnRenamed('WHEELS_OFF', 'wheels_off')\
                          .withColumnRenamed('WHEELS_ON', 'wheels_on')\
                          .withColumnRenamed('CRS_ARR_TIME', 'target_arr_time')\
                          .withColumnRenamed('ARR_TIME', 'actual_arr_time')\
                          .withColumnRenamed('ARR_DELAY', 'arr_delay')\
                          .withColumnRenamed('CANCELLED', 'cancelled')\
                          .withColumnRenamed('DIVERTED', 'diverted')\
                          .withColumnRenamed('CANCELLATION_CODE', 'cancellation_code')\
                          .withColumnRenamed('CARRIER_DELAY', 'carrier_delay')\
                          .withColumnRenamed('WEATHER_DELAY', 'weather_delay')\
                          .withColumnRenamed('NAS_DELAY', 'nas_delay')\
                          .withColumnRenamed('SECURITY_DELAY', 'security_delay')\
                          .withColumnRenamed('LATE_AIRCRAFT_DELAY', 'late_aircraft_delay')\
                          .withColumn('delay_label',F.when((F.col('actual_dep_time').isNull()) | (F.col('dep_delay') > 15.0), F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('day',F.date_format(F.col('fl_date'),'d'))\
                          .withColumn('month_day',F.concat_ws('-',F.col('month'),F.col('day')))\
                          .withColumn('holiday',F.when(F.col('month_day').isin(holidays),F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('spring',F.when((F.col('month') >=3) & (F.col('month') < 6), F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('summer',F.when((F.col('month') >=6) & (F.col('month') < 9), F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('fall',F.when((F.col('month') >=9) & (F.col('month') < 12), F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('winter',F.when(F.col('month').isin([12,1,2]), F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('depart_hour', extract_hour_udf(F.col('target_dep_time')))\
                          .withColumn('early_morning',F.when(F.col('depart_hour').isin([5,6,7]),F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('morning',F.when(F.col('depart_hour').isin([8,9,10,11]),F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('afternoon',F.when(F.col('depart_hour').isin([12,13,14,15,16]),F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('evening',F.when(F.col('depart_hour').isin([17,18,19,20]),F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('night',F.when(F.col('depart_hour').isin([12,22,23,24,1,2,3,4]),F.lit(1)).otherwise(F.lit(0)))\
                          .withColumn('busy_airport',F.when(F.col('airport_iata').isin(['ORD','ATL']),F.lit(1)).otherwise(F.lit(0)))\
                          .drop('day','month_day','depart_hour')\
                          .withColumn('tail_key',F.concat_ws(' ',F.col('fl_date'),F.col('tail_num')))

timedelta_2_hour = datetime.timedelta(minutes=120)
@udf(returnType=T.StringType())
def ymdh_floor(dt, adjust=timedelta_2_hour):
  d = dt - adjust
  return f'{d.year}{d.month:02d}{d.day:02d}{d.hour:02d}'

df_flights_p = df_flights_p.withColumn('target_dep_ymdh_n2', ymdh_floor(df_flights_p.target_dep_utc_timestamp))

print(df_flights_p.count())

display(df_flights_p)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1st Join Flights + (Weather + Airport + Weather Station)
# MAGIC 
# MAGIC With the updated weather dataset now containing IATA codes which are used to link the information to the flights dataset, the flight dataset needed additional cleanup and information before joining.  As described in the previous sections, the flight dataset was reduced by selecting only EDA identified columns.  Then the various times related to scheduled departure, actual departure, takeoff, landing, and arrival to the gate times were converted to UTC and added as additional columns.  Feature engineering added additional information such as seasons, stage of day (morning, evening, etc) as well as others (See Feature Engineer section for more details) A time adjusted key based on two hours prior to the scheduled departure time, floored by hour, was generated in the format YYYYMMDDHH. After review of the data, duplicate rows were detected and a drop operation was performed.  Having completed a majoring of cleaning and pruning of the flight dataset, the processed weather dataset was joined with the flight dataset using the IATA code and YYYYMMDDHH datasets, linking the flight data with weather information two hours prior to scheduled departure time.

# COMMAND ----------

df_flights_p.printSchema()
print(df_flights_p.count())
print(df_flights_t.columns)

# COMMAND ----------

df_flights_p = df_flights_p.drop_duplicates()
df_flights_p.count()

# COMMAND ----------

# Uncomment to rewrite the parquet

# df_flights_p.write.mode('overwrite').parquet(f"{blob_url}/df_flights_p{data_suffix}")

df_flights_p = spark.read.parquet(f"{blob_url}/df_flights_p")

# COMMAND ----------

df_station_weather_p.cache()
df_flights_p.cache()

# COMMAND ----------

df_flight_weather_p = df_flights_p.join(df_station_weather_p, (df_station_weather_p.weather_ymdh == df_flights_p.target_dep_ymdh_n2) & (df_station_weather_p.airport_iata == df_flights_p.airport_iata), "inner")
display(df_flight_weather_p)

# COMMAND ----------

df_flight_weather_p = df_flight_weather_p.drop(df_station_weather_p.airport_iata)

# COMMAND ----------

df_flight_weather_p = df_flight_weather_p.drop(df_station_weather_p.airport_iata)

# COMMAND ----------

Command took 0.57 seconds -- by judezhu@berkeley.edu at 7/25/2022, 8:20:30 PM on team16
# print('Dimension of flights: ({}, {})'.format(df_flights_p.count(), len(df_flights_p.columns)))
# print('Dimension of airports/stations/weather: ({}, {})'.format(df_station_weather_p.count(), len(df_station_weather_p.columns)))
# print('Dimension of airports/stations/weather/flights: ({}, {})'.format(df_flight_weather_p.count(), len(df_flight_weather_p.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Normalized the numeric Columns 

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
                    'daily_snow_fall']
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


# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA on Post 1st Join

# COMMAND ----------

scaled_col = [c+'_scaled' for c in numeric_col]
feature_pd = (df_flight_weather_p.select(*scaled_col).sample(False,0.01,81)).toPandas()

# COMMAND ----------

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
# MAGIC ## PCA

# COMMAND ----------

#selecting columns of features to be used in model training
feature_col = ['cancelled','diverted','carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay','holiday',
 'spring','summer','fall','winter','early_morning','morning','afternoon','evening','night','busy_airport'] + [c+'_scaled' for c in numeric_col]+['delay_label']

feature_table = df_flight_weather_p.select([col for col in feature_col])
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
# MAGIC ## Basic Logistic Regression Modeling

# COMMAND ----------

img_BASE_DIR = "dbfs:/FileStore/shared_uploads/jthsiao@berkeley.edu/model_pipeline_explored.jpg"
display(dbutils.fs.ls(f"{img_BASE_DIR}"))
zero_df = spark.read.format("image").load(img_BASE_DIR)
display(zero_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Splitting (Train/Test Sets)

# COMMAND ----------

df_final =  df_feature.withColumn('year', split(df_feature['fl_date'], '-').getItem(0))
df_final_test = df_final.filter(col('year') == "2021")
df_final_train = df_final.filter(col('year') < "2021")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Selection

# COMMAND ----------

# select the meaningful independent variables and dependent variable
features = ['quarter', 'month', 'day_of_week','carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay','holiday','spring','summer', 'fall','winter','early_morning','morning','afternoon','evening','night','busy_airport','hourly_dry_bulb_temperature_scaled','hourly_precipitation_scaled','hourly_relative_humidity_scaled','hourly_visibility_scaled','hourly_wind_gust_speed_scaled', 'hourly_wind_speed_scaled','daily_precipitation_scaled','daily_snow_depth_scaled','daily_snow_fall_scaled']
label = ["delay_label"]
df_final_test =  df_final_test.select(*features, *label)
df_final_train =  df_final_train.select(*features, *label)

# COMMAND ----------

display(df_final_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression Modeling

# COMMAND ----------

# assemble the input features as one vector
assembler = VectorAssembler(inputCols=features,outputCol="features") 
# define the LM model with basic parameters as ridge regression model
lr = LogisticRegression(featuresCol="features", labelCol="delay_label", regParam=1.0)
# build the pipeline and create the model
pipeline = Pipeline(stages=[assembler, lr])
model = pipeline.fit(df_final_train)

# COMMAND ----------

# predict for training set and testing set
prediction_train = model.transform(df_final_train)
prediction_test = model.transform(df_final_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Domain Specific Metrics (ROC curve) for Training/Test Set

# COMMAND ----------

# draw the roc curve for training set
display(model.stages[-1], prediction_train.drop("prediction", "rawPrediction", "probability"), "ROC")

# COMMAND ----------

# draw the roc curve for testing set
display(model.stages[-1], prediction_test.drop("prediction", "rawPrediction", "probability"), "ROC")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Standard Metrics for Training/Test Set

# COMMAND ----------

# establise the evaluators
recall = MulticlassClassificationEvaluator(metricName="weightedRecall", labelCol="delay_label")
precision = MulticlassClassificationEvaluator(metricName="weightedPrecision", labelCol="delay_label")
f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="delay_label")

# COMMAND ----------

# evaluation metics for training data
print(f"Weighted Recall: {recall.evaluate(prediction_train)}")
print(f"Weighted Precision: {precision.evaluate(prediction_train)}")
print(f"F1: {f1.evaluate(prediction_train)}")

# COMMAND ----------

# evaluation metrics for testing data
print(f"Weighted Recall: {recall.evaluate(prediction_test)}")
print(f"Weighted Precision: {precision.evaluate(prediction_test)}")
print(f"F1: {f1.evaluate(prediction_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC As we can see from the ROC curves of the training set (2015 - 2020 data) and testing set (2021 data), our classifier gives curves relatively close to the top-left corner which indicates a good performance of our model. As a baseline, a random classifier is expected to give points lying along the diagonal (FPR = TPR). The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test. That means our classifier is providing a relatively accurate test result based on the curves we have seen.  <br>
# MAGIC The F1_scores of our model are 0.74014 (training set) and 0.74095 (testing set), the scores look pretty satisfactory as they are generated based on a draft logistic regression model without hyper-tuning and cross-validation on Time Series.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Open Issues/Next Steps for Phase 3
# MAGIC 
# MAGIC - Incorporate advance model libraries
# MAGIC - Cross validation on time series
# MAGIC - Category variable indexing for LM model
# MAGIC - Hyper-tuning and cross-validation on a huge dataset is very time-consuming - is there a better way to chose the optimized hyper-parameters without run through training process over and over again? 
