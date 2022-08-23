# Databricks notebook source
# MAGIC %md
# MAGIC # Accufly Final Report
# MAGIC 
# MAGIC **Team Members:** Jerry Hsiao, Chi Ma, Fidelia Nawar, Jude Zhu

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abstract
# MAGIC 
# MAGIC The flight industry plays a crucial goal in the world’s transportation sector, and many businesses and people rely on various airlines to connect to other parts of the world. Unfortunately, flight delays have gradually increased and become inevitable due to weather conditions or operational issues, incurring significant financial losses to airline companies and customers. To resolve this issue, our team built a model to accurately predict flight delays, with our primary customers being airline companies. We define a delay as a 15-minute delay (or greater) from the original departure time, and the prediction should be performed at least 2 hours before departure time. We will use the information available 2 hours before the scheduled flight departure time to predict if the flight will be >15 minutes late for departure (including cancellation) or not. We utilized several datasets, including weather, station, flight date, time, and holidays and performed feature engineering to rank airports using PageRank. We experimented with three hyperparameter-tuned models: linear regression, Random Forest, and Tree Ensembles. The best final model for this project was the hyper-tuned linear regression model, reaching an AUC score of 0.60838, and Random Forest and Tree Ensembles AUC scores falling closely behind at 0.59900 and 0.57904, respectively.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Case
# MAGIC According to a 2020 study, the average delay costs about $92,000 per flight, and the drivers of cost are fuel, ground personnel, maintenance, aircraft ownership, and the need for extra gates, all of which impose costs on airline customers (including shippers) in the form of lost productivity, wages, and goodwill. A high-performance prediction model that can accurately predict if a flight will be delayed by >15 minutes of its scheduled departure time is precious to the passengers and the flight carrier. For passengers, if a flight delay is known 2 hours before its scheduled departure, they can make alternative plans instead of rushing through airport traffic. For carriers, a reliable prediction is more valuable as the company can take proactive actions to rearrange crew, make plans with airport gate control, and arrange flight change or compensation plans with passengers ahead of time. However, such predictions must be of high accuracy, or they will adversely affect both the passenger and the carrier.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics
# MAGIC 
# MAGIC The evaluation metrics used in our project are the following:
# MAGIC - Presicion: $${\(\frac {Actual Delay} {All Predicted Delay}\)}$$
# MAGIC - Recall: $${\(\frac {Delays Predicted} {All Delays}\)}$$
# MAGIC - F1 Score $${\(\frac {2 * Precision * Recall} {Precision + Recall}\)}$$
# MAGIC - AOC: Area under the ROC curve
# MAGIC 
# MAGIC Since the focus of our model is to reduce costs for airlines, we want to focus on minimizing the rate of false positives. In the context of our project, we want to reduce the rate at which flights are predicted to be delayed when they are not actually delayed, which would help minimize airline costs. We use the measurement of precision since it is heavily influenced by minimizing the rate of false positives. We also consider recall to be able to measure delays when they actually occur. We also incorporate F1 score, which sums up the predictive performance of a model by combining precision and recall. We chose to use F1 instead of accuracy because of imbalanced training data. 
# MAGIC 
# MAGIC We also incorporate Area Under the ROC curve (AUC) to evaluate how well our model classifies positive and negative outcomes at all possible cutoffs. The AUC will be used to evaluate how well our logistic regression model classifies positive and negative outcomes at all possible cutoffs. It can range from 0.5 to 1, and the larger it is the better. <br>
# MAGIC 
# MAGIC For the gradient boosted tree, we have two loss functions, i.e: <br>
# MAGIC A custom loss function that we calculate the gradient for: \\(L(y_{i}, \hat{y_{i}})\\) <br>
# MAGIC The loss function used by the tree that fits the gradient, which is always squared loss. <br>
# MAGIC 
# MAGIC For the logistic regression model, L2 Regularization will be used to solve the problem of overfitting our model by penalizing the cost function. It does so by using an additional penalty term in the cost function.
# MAGIC \\(J\lgroup w \rgroup = \frac{1}{m} \displaystyle\sum\limits_{i=1}^m Cost(h(x^{i}), h(y^{i})) +\frac{\lambda}{2m}\displaystyle\sum\limits_{j=1}^n w_j^2\\)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Project Pipeline
# MAGIC ![Pipeline](files/tables/Screen_Shot_2022_08_04_at_10_46_12_AM.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Supporting Data
# MAGIC ##### Flights
# MAGIC This dataset lists passenger flight's on-time performance data taken from the TranStats data collection from the U.S. Department of Transportation (DOT) and contains flight information from 2015-2021. Reporting carriers are required to (or voluntarily) report on-time data for flights they operate: on-time arrival and departure data for non-stop domestic flights by month and year, by the carrier, and by origin and destination airport. The dataset includes scheduled and actual departure and arrival times, canceled and diverted flights, taxi-out and taxi-in times, causes of delay and cancellation, air time, and non-stop distance. This dataset will be used to track which flights were marked as delayed (departure) and collect information on the date, time, airline, departure city, etc., for each flight. 
# MAGIC 
# MAGIC ##### Weather
# MAGIC The weather data table includes weather data corresponding to the origin and destination airports at the time of departure and arrival, respectively. It was downloaded from the National Oceanic and Atmospheric Administration repository and contains weather information from 2015 to 2021. The dataset comprises summaries from major airport weather stations that include a daily account of temperature extremes, degree days, precipitation amounts, and winds, along with hourly precipitation amounts and abbreviated 3-hourly weather observations. This dataset will be used in conjunction with the flight dataset to analyze the weather patterns on the days of delayed flights. 
# MAGIC 
# MAGIC ##### Station
# MAGIC The station data table provides metadata about each weather station, downloaded from the U.S. Department of Transportation. It includes details such as the longitude/latitude of the weather station, the ICAO code, station identifier, etc. This dataset will be used with the flights and weather dataset to link the two datasets together. 
# MAGIC 
# MAGIC ##### Airport
# MAGIC 
# MAGIC The station data table provides metadata about each airport, downloaded from [Airport Database](https://www.partow.net/miscellaneous/airportdatabase/index.html#Downloads). It includes details such as the longitude/latitude of the airport, airport name, IATA airport code, ICAO code, etc. This dataset will be used with the station and weather datasets to link weather data to airport.
# MAGIC 
# MAGIC ##### US Holiday Days
# MAGIC 
# MAGIC The [U.S. holiday dataset](https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021?resource=download) contains dates of U.S. holidays from 2004-2021. We will join this to the flights dataset to identify U.S. holidays (Kaggle 2020)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA
# MAGIC The data for the model training consisted of four sources - weather data from 2015 to 2021,  weather station information, airport information, and flight data from 2015 to 2021.  The data was joined to produce a baseline dataset containing weather data two hours prior for each corresponding scheduled departing flight.
# MAGIC 
# MAGIC ### Flights
# MAGIC There are identifiable trends in flight delays among the entire dataset of flight information. For the interest of the classification outcome (delay or not), the average number of delays seem to be highest on Monday and Sunday, and less frequent number of delays on Tuesday, Wednesday, and Fridays.
# MAGIC 
# MAGIC | ![delays by day](/files/shared_uploads/jthsiao@berkeley.edu/delays_by_day.png) |
# MAGIC | -- |
# MAGIC | Delays by Day from 2015 to 2021 |
# MAGIC 
# MAGIC | ![delays by month](/files/shared_uploads/jthsiao@berkeley.edu/delays_by_month.png) |
# MAGIC | -- |
# MAGIC | Delays by Month from 2015 to 2021 |
# MAGIC 
# MAGIC Based on the graph above, the summer months seem to have the highest average number of departure delays (June, July, and August), in addition to December also having a relatively high number of average delays. This may be due to more people traveling during December for holidays, and also during the summer, which could cause an increase in the amount of flights and thereby increasing the amount of delays in proportion to the other months.
# MAGIC 
# MAGIC These observations are important in determining what could be the useful feature to predict flight delay. These bivariate analysis leads us to add additional features such as ***season***, ***is_holiday***, ***part of day***
# MAGIC 
# MAGIC <img src="/files/tables/Screen_Shot_2022_08_03_at_12_29_19_AM.png" width="550">
# MAGIC 
# MAGIC From the bar chart above, we can see the distribution of delayed or not delayed flights. About 78% of the final joined table consists of flights that are not delayed, whereas about 22% of flights are delayed. For the simplicity of our project, we chose to impute canceled flights, as explained in feature engineering. This proportionality is vital since understanding the class imbalance is necessary to achieve high levels of prediction ability. To reduce the effects of class imbalance and improve our model's predictive abilities, we employed undersampling on our dataset.
# MAGIC 
# MAGIC | Correlation Matrix Prior to Feature Selection | Correlation Matrix Post Feature Selection |
# MAGIC | -- | -- |
# MAGIC | <img src="/files/tables/Screen_Shot_2022_08_03_at_8_25_37_PM.png" width="600"> | <img src="/files/tables/Screen_Shot_2022_08_03_at_8_25_42_PM.png" width="600"> |
# MAGIC 
# MAGIC We have a correlation matrix of our weather features on the left before feature selection. We can see how variables like snow_fall/snow_depth are heavily correlated with daily_precipitation and other weather factors, with a coefficient of about 1.0. This means that including these features for our model predictions is unnecessary, so we chose to reduce the feature list by removing snowfall information while retaining the same data. We decided to include the features with lower correlation coefficients not to skew the model predictions. We only had the most critical components, as shown in the second correlation matrix to the right, post feature selection.
# MAGIC 
# MAGIC ![weather features analysis](files/tables/Screen_Shot_2022_08_03_at_8_18_44_PM.png)
# MAGIC 
# MAGIC Looking at the charts above, we can see the general distribution of important weather features after removing the highly correlated ones, such as daily_precipitation, relative_humidity, and dry_bulb_temperature. For some features such as hourly_precipitation, hourly_visibility, hourly_wind_speed, etc. it seems like the distribution is centered around 0, which might be due to the data imputation. 
# MAGIC 
# MAGIC ![bivariate analysis](files/tables/Screen_Shot_2022_08_05_at_4_51_10_PM.png)
# MAGIC 
# MAGIC Looking at the bivariate charts above, we can see a side by side comparison of frequency of delays in different times of day, early_morning, morning, afternoon, and evening. The graphs show that there seem to be greater number of delays in the afternoon/evening, which would be important information to consider for airlines as they prepare their operations to handle delays. In the box plot below, we can see how delayed flights had greater wind speeds compared to other weather features.
# MAGIC 
# MAGIC ![weather bivariate](files/tables/Screen_Shot_2022_08_05_at_4_54_14_PM.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC ### Data Transformation
# MAGIC - Standardize Date_time format to make up primary and foreign keys for join. That’s being done by converting the departure time to UTC timestamp and create hourly windows of interest for joining weather data
# MAGIC - Add delay_labels based on the departure time delay. 1 for >=15 minutes departure delay and 0 otherwise. Note: Canceled flights are labeled as "1"
# MAGIC 
# MAGIC ### Additional Features
# MAGIC - Added season categories, holiday indicators, day part categories, and airport rank using a pagerank algorithm. Used one hot encoding to for categorical variables
# MAGIC   - ***PageRank***: Since airplane flight path actually form a node-link graph, with each airport as a node and a flight in between as an edge. We thought it would be beneficial to use PageRank analogy and calculate the importance of each airport, and see if such rank score has any effect in predicting flight delay. There will be no dangling nodes in this airport network
# MAGIC 
# MAGIC     Top 10 Airports from 2015 to 2021 by PageRank
# MAGIC     
# MAGIC     | Rank | Airport |
# MAGIC     |----------|--------------|
# MAGIC     | 1        | ATL          |
# MAGIC     | 2        | ORD          |
# MAGIC     | 3        | DFW          |
# MAGIC     | 4        | DEN          |
# MAGIC     | 5        | LAX          |
# MAGIC     | 6        | PHX          |
# MAGIC     | 7        | LAS          |
# MAGIC     | 8        | CLT          |
# MAGIC     | 9        | SFO          |
# MAGIC     | 10       | IAH          |  
# MAGIC 
# MAGIC   - Transforming the flight departure iata and flight destination iata into a graph, the flight rank is calculated using 15 iterations and added to the dataset as an additional feature
# MAGIC   - ***Holiday indicator***: Using an additional US holiday dataset, we are able to identify US holidays for all the flight dates in our dataset. Since airports don't tend to be busy on the day of the US holiday, but rather the day before and after US holiday, we indicated +-1 day of US holiday as holiday
# MAGIC 
# MAGIC ### Scaling/Normalization
# MAGIC - For numeric columns, we will use Z-score normalization so that when we run algorithms using gradient descent, we won't run into non-convergence. The formula for normalizing is:
# MAGIC $${\(\frac {X_i - \bar{X}} s\)}$$
# MAGIC - Numeric Columns we decided in to keep in the final features 
# MAGIC 
# MAGIC   
# MAGIC    |hourly_dry_bulb_temperature	   |hourly_precipitation   |hourly_relative_humidity   |hourly_visibility  |hourly_wind_speed  |daily_precipitation	   |dep rank|
# MAGIC    |-------------------------------|-----------------------|---------------------------|-------------------|-------------------|-----------------------|--------|
# MAGIC - Using a standard scaler, normalization is done on each of the numeric columns
# MAGIC - Note the normalization is done after ***undersampling***
# MAGIC 
# MAGIC ### Missing Data Management
# MAGIC 
# MAGIC - We did not need to handle all missing data or NaN in the beginning, because many columns were dropped in later phases. 
# MAGIC - After finalizing the columns to keep, we are left with NaN and nulls in some of the weather columns, so data imputation was implemented to fill those with 0
# MAGIC   - The weather dataset and many fields populated with nulls.  A majority of the columns of weather data were identified as not useful such as yearly and monthly metrics, backup related columns, etc.
# MAGIC   * 'hourly_wind_gust_speed': Spikes in wind speeds above 9 knots lasting less than 20 are classified as gusts, else the column is null. Converted nulls to 0. 
# MAGIC   * 'daily_precipitation', 'daily_snow_depth', 'daily_snow_fall': Daily metrics which appear contain many null values.  This could be because the data is not reported.  Converted the nulls to 0 which would be equivalent to these features not affecting a flight.
# MAGIC - For flights, the majority of flight data was well populated. There were flights that had departure times and arrive times populated with nulls. Upon further investigation, these were because the flights were canceled. In this case, canceled flights were imputed as a factor to feature 'delay_label' for modeling. The columns that were null because of cancellations were not used as features.
# MAGIC 
# MAGIC 
# MAGIC ### Feature Selection
# MAGIC - After removing redundant columns (such as columns used to make the primary and foreign keys for data joining), we are left with the following columns as potential features for ML
# MAGIC - Numeric Features: 'hourly_dry_bulb_temperature', 'hourly_precipitation', 'hourly_relative_humidity','hourly_visibility','hourly_wind_gust_speed', 'hourly_wind_speed', 'daily_precipitation', 'daily_snow_depth', 'daily_snow_fall', 'dep_rank'
# MAGIC   - Among these, daily_precipitation, daily_snow_depth and daily_snow_fall appear to be highly correlated. Upon further research, daily_precipitation includes snowfall, so we are removing the snowfall related columns. Also removed hourly_wind_gust_speed as it highly correlates with hourly_wind_speed.
# MAGIC - Categorical Features: 'carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay','is_holiday','spring','summer','fall','winter','early_morning','morning','afternoon','evening','night'
# MAGIC   - features should be information available up to 2 hours before flight's scheduled departure time. So ['carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay'] should not be included in the features used for prediction
# MAGIC - The remaining features contain only information availalable 2 hours before flight's scheduled departure time
# MAGIC 
# MAGIC   - ***Numeric Features***
# MAGIC    |hourly_dry_bulb_temperature    |hourly_precipitation   |hourly_relative_humidity   |hourly_visibility  |hourly_wind_speed  |daily_precipitation	   |dep rank|
# MAGIC    |-------------------------------|-----------------------|---------------------------|-------------------|-------------------|-----------------------|--------|
# MAGIC   
# MAGIC   - ***Categorical Features***
# MAGIC    |is_holiday|spring|summer|fall|winter|Early morning|morning|afternoon|evening|night|
# MAGIC    |----------|------|------|----|------|-------------|-------|---------|-------|-----|

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Join
# MAGIC In general, the data joining steps were as follows:
# MAGIC 
# MAGIC To join the weather and flight datasets together, additional information was required. In their raw state there was no common key to link the two datasets together. The weather dataset contained a “station id”, which is the station identifier of the station that collected the measurements. The flight dataset contained the “origin” which is the IATA code of the departing flight. To link the two datasets, a weather station and airport dataset was introduced to bridge the gap i.e. Weather <-> Weather Station <-> Airport <-> Flights.
# MAGIC   
# MAGIC ***Airport Code + Weather Station***
# MAGIC   - After reviewing the weather station dataset, which contained station information and neighboring station information. Since the station information was missing the ICAO code, the neighboring station information was utilized as it contained the ICAO code and station identifier. By leveraging the Airport Database, containing ICAO and IATA codes, the two datasets were joined via ICAO code as the key to provide a new table containing ICAO codes, IATA codes, and station identifiers. (Partow, 2017) To help reduce the dataset size duplicate rows were dropped.
# MAGIC       
# MAGIC ***Weather + (Airport + Weather Station)***
# MAGIC   - After obtaining the Airport + Weather Station dataset, the IATA codes were added to the weather dataset via a join operation on station identifier. The dataset size was reduced by first removing the columns identified for pruning from EDA activities; then duplicate rows were dropped. Nulls in specific columns were converted to zeros in cases identified to make sense such as hourly_wind_gust_speed which would be set to null when the current wind speed was not classified as a gust. Additional time related columns were added, utc timestamp and a time adjusted key rounded to the nearest hour in the format YYYYMMDDHH. Multiple rows per station within the same hour were aggregated based on various techniques such as selecting the max value for precipitation or the mean value for humidity.
# MAGIC       - Weather Aggregation: After having limited the number of weather features and performing time conversions and creation of special time based key, we aggregated the weather information per hour at each station. The aggregation depends on the type of information such as for temperature the mean is taken where as wind speed the max value as a high wind value could trigger weather condition delay of a flight. Daily metrics were aggregated as max although in some cases these values would not change much within an hour and are used to identify situations where the airport is impacted due to poor weather conditions throughout the day.
# MAGIC 
# MAGIC  ***Flights + (Weather + Airport + Weather Station)***
# MAGIC    - With the updated weather dataset now containing IATA codes which are used to link the information to the flights dataset, the flight dataset needed additional cleanup and information before joining. As described in the previous sections, the flight dataset was reduced by selecting only EDA identified columns. Then the various times related to scheduled departure, actual departure, takeoff, landing, and arrival to the gate times were converted to UTC and added as additional columns. Feature engineering added additional information such as seasons, stage of day (morning, evening, etc) as well as others (See Feature Engineer section for more details) A time adjusted key based on two hours prior to the scheduled departure time, floored by hour, was generated in the format YYYYMMDDHH. After review of the data, duplicate rows were detected and a drop operation was performed. Having completed a majoring of cleaning and pruning of the flight dataset, the processed weather dataset was joined with the flight dataset using the IATA code and YYYYMMDDHH datasets, linking the flight data with weather information two hours prior to scheduled departure time.
# MAGIC 
# MAGIC ***Airport PageRank***
# MAGIC   - After the each airport's Pagerank has been generated, the rank score is then joined to the main dataset using departure IATA code.
# MAGIC  
# MAGIC ***US Holiday***
# MAGIC   - In order to more accurately identify US holidays, we obtained an additional dataset from Kaggle that contains the dates of US holidays from 2004-2021. Since we think the day before and after holiday tend to affect air travel(sometimes more than the holiday date itself), we tweaked the holiday dataset to refelect that +- day of holiday as holiday too. Finally, this dataset is joined onto the main dataset on flight date. 
# MAGIC 
# MAGIC 
# MAGIC The initial join (Flights+Weather data) took about 2 hours to join, and we noticed caching datasets prior to joining significant sped up the runtime. The runtimes of the other two joins
# MAGIC (pagerank, US holidays) were insignificant.
# MAGIC 
# MAGIC To see the table joining code in further detail, please visit our [Phase 2 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/3326810404140495/command/3326810404140496).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experimental Evaluation
# MAGIC For our final model, we attempted three methods to generate a classification prediction model - LinearRegression, Random Forest, Tree Ensembles (XGBoost).  Various cross validation methods were also utilized such as random K fold and rolling window by year.

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA - Logistic Regression
# MAGIC 
# MAGIC Logistic regression estimates the probability of an event occurring, whether it will be flight delays or not in our scenario, based on a given dataset of independent variables. As the first step, we used principal component analysis to decompose the features into five principle components and ran a Logistic Regression on it with the hyperparameters below. The resulting AOC is 60.8%. And based on the evaluation output we already have, we perform cross-validation on Time Series and hyper tuning.
# MAGIC As the original training dataset is enormous, it takes a long time to cross-validate and hyper-tune the model based on the original dataset. To increase the time efficiency, we sample the training dataset to perform cross-validation and hyper tuning and collect the hyperparameters of the best model to train the final model with a full training dataset. Then, we use the model to predict and evaluate the full testing dataset.
# MAGIC The cross-validation we perform is a custom version of cross-validation, which will iterate over a rolling basis. We also include a categorical indexer, vector assembler, and pca model in our model pipeline when training the models.
# MAGIC 
# MAGIC We tuned the logistic regression model with the following parameters:
# MAGIC - elasticNetParam: determines the penalty level of our model. With an alpha = 0, the penalty is an L2 penalty. For an alpha = 1, it is an L1 penalty.
# MAGIC - regParam: the regularization parameter of our model that helps generalize better on unseen data by preventing the algorithm from overfitting the training dataset
# MAGIC - maxIter: the maximum number of iterations taken for the solvers to converge.
# MAGIC  <br>
# MAGIC   
# MAGIC **Table of Best Model Parameters**
# MAGIC | Parameter | Value | Observation | 
# MAGIC | -- | -- | -- |
# MAGIC | regParam | 0.03 | 0.03 regularization parameter helps generalize better on unseen data |
# MAGIC | maxIter | 10 | The maximum number of interations to converge is 10 |
# MAGIC | elasticNetParam | 0 | L2 penalty is preferred |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest
# MAGIC 
# MAGIC Random forest is a Supervised Machine Learning Algorithm that is used widely in Classification and Regression problems. It builds decision trees on different samples and takes their majority vote for classification.
# MAGIC 
# MAGIC The cross validation we perform is a custom version of cross validation which will iterate over a rolling basis. We also include categorical indexer, vector assembler, and pca model in our model pipeline when training the models.
# MAGIC   
# MAGIC **Table of Best Model Parameters**
# MAGIC | Parameters | Value | Observation | 
# MAGIC | -- | -- | -- |
# MAGIC | bootstrap | True | Bootstrap samples are used when building trees |
# MAGIC | numTrees | 40 | 40 trees in the random forest. |
# MAGIC | maxDepth | 5 |  maximum depth of tree is 5 |
# MAGIC | featureSubsetStrategy | auto | if numTrees == 1, set to “all”; if numTrees > 1 (forest) set to “sqrt” |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tree Ensembles
# MAGIC 
# MAGIC The model generatered by tree emsembles leveraged the XGBoost library. Various hyperparameters were investigated - depth, trees, learning rate, L1 (alpha) and L2 (lambda) regularization, gamma (min loss for branching), and various tree methods.  Given execution time and limited cluster capability, a minimal set of each parameter was utilized through GridSearch. Evidentaily, moving to an  hand-tuned process proved a more fruitful result generation.  There was a execution hard limit at 100 trees (n_estimators), which was empirically found. See the Appendix for the various experiments and parameters explored.  The table below shows the best model generated using F1 score as the metric. 
# MAGIC 
# MAGIC We tuned the tree ensembles model with the following parameters:
# MAGIC - Depth and # of Trees: selected as key parameters because of their potential affect to performance.  With more trees, there was the assumed possibility of increasing likelihood of greater aggreement within an ensamble.  However, at higher number this appeared to not be the case.  
# MAGIC - L1 and L2: selected to regularize the model (penalize complexity)
# MAGIC - Gamma: selected to control partitioning of leaf nodes, which was found at higher values (more conservative) to improve the model slightly.
# MAGIC   
# MAGIC **Table of Best Model Parameters**
# MAGIC | Parameter | Value | Observation | 
# MAGIC | -- | -- | -- |
# MAGIC | Depth | 6 | 3, 10, 15 negative impact |
# MAGIC | # Trees | 50 | 10 negative impact, 70, 90 slight negative impact, >= 100 unable to run |
# MAGIC | Learning Rate | .1 | 1e-3, 1 negative impact |
# MAGIC | L1 | .1 | 1e-3 slight negative impact |
# MAGIC | L2 | 1e-5 | 1e-3, 1e-1, 1, 2, 3 nearly no effect |
# MAGIC | Gamma | 7 | 1e-3, 1, 5 no/slight negative effect |
# MAGIC | Tree Method | approx | `hist` slight negative effect |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Results
# MAGIC 
# MAGIC The table below shows the best results from each method - blind test results, training and validation. Include an experiment name and/or short description of the model,  number of features used.
# MAGIC 
# MAGIC **Table of Best Model Scores**
# MAGIC | Model Method | AUC | Recall | Precision | F1 | Runtime (minutes) |
# MAGIC | - | - | - | - | - | - |
# MAGIC |Logistic Regression |0.60838 | 0.58476 | 0.58476 | 0.58475 | 3.43 |
# MAGIC | Random Forest | 0.59900 | 0.56834 | 0.56864 | 0.56793 | 9.20 |
# MAGIC | Tree Ensemble | 0.57904 | 0.51704 | 0.62915  | 0.38410 | 4.06 |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Results Analysis
# MAGIC We are basing our model performance on AUC, as the Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve. We should use it when we care equally about positive and negative classes. It naturally extends the imbalanced data discussion from the last section. Since we care about true negatives as much as we care about true positives in the Accufly scenario, it makes sense to use ROC AUC. <br>
# MAGIC 
# MAGIC As we can see from the test results in the section above, the logistic regression model has an AUC score of around 0.61, random forest model has an AUC score of approximately 0.60, and the tree ensemble model has an AUC score of about 0.58. The AUC scores for each model are relatively close, where the LR model has the highest AUC score. As the dependent variable is binary in our scenario, and the number of observations is much larger than the number of features, it is appropriate to use logistic regression as our flight delay prediction model. Also, logistic regression analysis is valuable for predicting the likelihood of an event or flight delay in this case. It helps determine the probabilities between delay and not delay classes. That makes the logistic regression model the best fit in our scenario. <br>
# MAGIC 
# MAGIC Please refer to [Phase 4 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/3326810404140499/command/849916349912589), for model code/implementation and ROC curves.
# MAGIC 
# MAGIC ### Data Leakage
# MAGIC 
# MAGIC Data leakage is a problem in machine learning when developing predictive models. Data leakage occurs when information from the training dataset is used to create the model. This additional information can allow the model to learn or know something that it otherwise would not know and invalidate the estimated performance of the constructed mode. Our initial baseline logistic regression model had data leakage, as we included the delay features in our training dataset (security delay, weather delay, carrier delay, etc.) These delay features are post-predict data correlated to our dependent variable delay_label. In our cause, including those delay features performed the suspiciously high evaluation scores when testing our baseline model, so we removed those features during feature selection for our final model to avoid data leakage.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Novel Approaches
# MAGIC 
# MAGIC **Indicator Variables**<br>
# MAGIC A novel approach for feature selection was adding indicator variables such as season, time of day (early_morning, morning, afternoon, evening, night), and holidays. Including these indicator variables helped refine our model since these features are highly predictive of flight delays. For example, if there is higher traffic at airports in the morning or the night before holidays, there is a greater chance of delays. We can create a more accurate model to predict delays by incorporating these new features.
# MAGIC 
# MAGIC **PageRank**<br>
# MAGIC One of the main novel approaches we took was implementing the PageRank algorithm. Since the airport network can be modeled as a graph, where each airport is considered a node and each edge is considered a flight path between airports, we can employ various graph algorithms to aid in our prediction process, such as measuring centrality. Centrality measures how "important" or "central" one node is compared to the others, and PageRank is an innovative algorithm used to obtain the centrality of nodes. This centrality measure is essential to our project because we can easily rank airports to identify which airports have higher pass-through traffic and help us predict potential problems in other airports in the network. 
# MAGIC 
# MAGIC PageRank (PR) is an algorithm created by Google Search to rank web pages in their search engine results. According to Google: PageRank works by counting the number and quality of links to a page to determine a rough estimate of the website's importance. The underlying assumption is that more important websites will likely receive more links from other websites. We apply this to our project to determine 'importance' within the network. This model can be seen as a Markov chain to predict the behavior of a system that travels from one state to another, considering only the current condition. In the context of the flight paths, the Markov chain allows us to model a plane that is randomly placed at any airport and choose its destination airport randomly. Still, fortunately, no dangling nodes exist in our network, making the problem easier. We can determine the probabilities of moving to each potential airport, represented by a transition matrix, and the matrix is learned based on connectivity between nodes. We can rank centrality with this algorithm because the matrix eventually converges to a steady state of probabilities. This model works based on two conditions: irreducibility (each airport must be able to be connected to all other airports) and aperiodicity (no sinks that would allow the plane to end up at the same airport continuously). We implemented these two conditions by applying adjustments such as teleportation, which allows the surfer (plane) to randomly jump to any other airport in the network and spread the probability of leaving a particular node across all other nodes (stochasticity).
# MAGIC 
# MAGIC This novel approach of using PageRank allows us to determine the "important" airports within the flight network and understand how their centrality would affect the frequency of flight delays.
# MAGIC 
# MAGIC More detailed PageRank implementation can be found in the [Phase 3 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/3326810404140497/command/28620905306066). 
# MAGIC 
# MAGIC **Caching**<br>
# MAGIC Initially, the main join (Flights + Weather) took over 5 hours, so we cached the data frames before joining, significantly reducing the runtime.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance and Scalability Concerns <br><br>
# MAGIC 
# MAGIC - The main join (flights + weather) and PageRank both took more than 1 hour to complete. A performance concern is that our code may not be the most efficient runtime-wise.
# MAGIC - The project has been an iterative process, especially during the Feature Engineering stage. The need to add on additional datasets (such as IATA <-> weather station as a bridging dataset or the US holiday dataset) creates additional work and requires several full dataset joins. This hinders scalability if such exercise needs to be run frequently and at scale. A separate data aggregation process might be necessary for cleaner, easier to handle data.
# MAGIC - Due to the need for a separate PageRank algorithm and the need for several dataset joins, as well as the nature of this project (divided into 4 phases where each latter phase improves on the previous phase), we are unable to build a complete full pipeline this time where any raw data can be fed into the pipeline for the best performing model to run. This could be achievable with the right time and resources.
# MAGIC - We have not tested our model on more history, and there is questions on how predictive our model would be for data further back or most recent. 
# MAGIC - The F1 score and AUC for our model evaluations are in the 60% range. While this score may not seem ideal, I think it contributes to the following factors:
# MAGIC   - The prediction is made 2 hours prior to the flight departure. 2 hours is a wide window for airport conditions and weather conditions to change. The prediction will certainly become more accurate as more information becomes available closer to the scheduled departure time
# MAGIC   - The majority of numerical variables are weather conditions, which are, in fact, less indicative of a flight delay, as we have discovered during final model predictions
# MAGIC   - Based on our experience, airport maintenance is a significant driving factor for flight delays, but we only have minimal information on that. Although an airport rank was calculated, it doesn't explain the complete picture of airport/gate maintenance
# MAGIC   - If we used information about the tail number, the result could improve. But given the complexity, we decide to leave it for future improvement
# MAGIC - Generalizability
# MAGIC   - We only split data into training and testing, not validation, so we are unsure about the generalizability of the models outside of our dataset. If we obtained data of long history, we may re-split the data and include a validation dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gap Analysis
# MAGIC  <br>
# MAGIC Based on the current scoreboard (Thursday 8/4 3:15am PST), the results of our updated models seems to be scoring relatively the same compared to our peers, around 0.6, albeit others used F0.5 or F2, whereas we used F1. Most other groups also seemed to have used the same models, Random Forest, ensemble, and linear regression. We do not see improvement in our metrics from last phase's evaluation, because we discovered that we had data leakage from our previous model implementation. For this phase, we believe we have achieved relatively reasonable metrics after our hyper parameter tuning and have a few points of consideration to improve our scores even further for future phases. Based on the fishbone diagram, we have some potential causes of “low” metrics that we can try to improve for the future: <br><br>
# MAGIC 
# MAGIC - <b>Feature Selection Techniques</b>: While we believe we identified the most relevant features for the model and trained on the meaningful independent/dependent variables, there is no guarantee that the features we selected were the “best” ones. It could be possible that we do not provide the model with enough features (or too many features), which would affect the final performance. In this scenario, we could use further PCA techniques to aid in choose the best features to ensure our model provides the most useful information. We also did not leverage features like COVID data or the previous flight delay tail attribute.
# MAGIC - <b>Data Analysis Decisions</b>: We are using z-score instead of min-max and standard scalar normalization. Min-max normalization guarantees that all features will have the exact same scale but does not handle outliers well, whereas z-score normalization handles outliers, but does not produce normalized data with the exact same scale.
# MAGIC - <b>Limited Performance Tuning</b>: Because the models take a decent amount of time to train/fit, it makes it difficult to be able to change many parameters frequently to analyze the results, especially when it comes to the cross validation on time series, the time cost could be significantly high. To address that, we sampled the full dataset and use the sampled dataset to do the CV + hyper tuning, and then used the best parameters to train the model with full dataset.
# MAGIC - <b>Algorithm Methodology</b>: For this phase, our CV hyper tuned model used time series cross validation on a rolling basis over the years, but we would be curious to explore if there is better performance if it were modeled over months or quarters. Since the CV over years took significant time to train, we were unable to model over months because it would be too computationally expensive. 
# MAGIC - <b>SMOTE in Scale</b>: Because SMOTE, an oversampling method, requires leveraging a method to group similar samples; a majority of implementations utilize kNN.  kNN for large datasets is extremely memory intensive and exceeded the clusters capabilities.  Another attempt utilizing locality sensitive hashing which also could not complete execution. Both implementations took had execution times in the order of hours prior to failing.  A new approach was found, [Approx-SMOTE](https://github.com/mjuez/approx-smote), which provides an approximated kNN approach which advertised 7-28x speed improvements, however the implementation is in Scala and because of limited time could not be explored.
# MAGIC 
# MAGIC ![Final Gap Analysis](files/tables/Final_Gap_Analysis_2.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC 
# MAGIC The focus of the project was to build a predictive model that can accurately predict flight delays up to 2 hours in advance. We predict that machine learning pipelines with custom features can accurately predict flight delays using the datasets provided because we can analyze past trends in weather conditions, reasons for delays, airports where there are more delays, and many other necessary details combined to feed into our model to make predictions off of. We believe our model can accomplish this task based on the following:
# MAGIC 
# MAGIC - Feature engineering techniques performed
# MAGIC - Use PCA analysis to extract the most relevant features for our model
# MAGIC - Use PageRank to calculate the importance of each airport and see if their rank score has any effect in predicting flight delays
# MAGIC - Performing hyperparameter tuning and cross-validation over time series for our logistic regression, Random Forest, XGBoost (Phase III), and Tree Ensembles models
# MAGIC - Finding the most optimal features and parameters to continue to tune our evaluation metrics
# MAGIC 
# MAGIC As we saw in our results, the best performance of our model is about 0.60 for the AUC, compared to 0.78 for the baseline model (with data leaks). We are satisfied with our feature selection choices and parameter optimizations for this phase since we could address the data leak and account for it. 
# MAGIC 
# MAGIC #### Business Case Application
# MAGIC On average, there are about 976,000 delayed flights per year in average (from 2015-2019), and our best model would be able to detect 570,726 potential flight delays (976k * recall) and correctly predict 333,738 of those flight delays (potential flight delays detected * precision). According to the FAA/Nextor, the estimated annual costs of delays (direct cost to airlines and passengers, lost demand, and indirect costs) in 2018 were about $28 billion and $8.4 billion in delayed flights operations, so by using our hyper tuned logistic regression model, airlines can save about $2.9 billion per year in flight operation costs (333,738 / 976,000 * 8.4b), which equates to saving about $8,689 per delayed flight (2.9b / 333,738).
# MAGIC 
# MAGIC Our initial objective was to precisely predict flight delays, where a delay is categorized as at least a 15-minute further departure time than the original. Through our feature selection, use of various data sets, and implementation of model algorithms, we have been able to construct a model that can help airlines predict flight delays with about 60% precision. This will benefit airlines by reducing costs such as organizational, maintenance, passenger reimbursements, etc., while helping them better streamline their processes using predictions from our best model.
# MAGIC 
# MAGIC #### Key Findings
# MAGIC Based on our feature selection, it was interesting that weather features were not as indicative of flight delays as we might have thought initially. We figured that by removing the highly correlated weather features, we would be able to extract more accurate predictive abilities, but it didn't seem to improve performance noticeably. Another key finding that we initially determined was that there is a significant impact of prior flight delays to flight departures later in the day, so we would recommend airlines to be prepared have resources available to amend flight delays earlier in the day to prevent compounding effects to flights later in the day. Undersampling might have also benefited us since we prevented overfitting our models by over-sampling, which would have included duplicated examples from the minority class in the training dataset. Overall, the Tree Ensembles and Random Forest models still yield reasonably high accuracy and similar performance to the linear regression model, so we recommend future researchers continue tuning these models.
# MAGIC 
# MAGIC #### Challenges and Constraints
# MAGIC Like most other groups, the main challenge we encountered was successfully performing machine learning at scale, as the course name suggests. Most individuals in our group have not built models on datasets of this size, so it was a long, iterative process that required a lot of patience when performing time-intensive operations such as the central table join or training on the models. Our primary constraint was insufficient computational power in the cluster to explore and implement more robust algorithms. There was also the time/resource constraint since training the models and fixing bugs took a significant time to run, and if the notebook were to crash or we spent hours training to receive poor performance, we would have to restart the process all over again. Despite these bottlenecks, we took the time to learn about each dataset, feature, and algorithm to make the most informed decisions in our model tuning.  
# MAGIC 
# MAGIC #### Future Work
# MAGIC We still have other aspects we want to explore for future work, such as breaking down the busiest airports per year and analyzing that data. We would also hope to refine our feature engineering to explore other relevant features such as airline reputation, airport maintenance, and the arrival of the tail number. Given the computational resources, we hope to finish implementing SMOTE in Scale or approximate SMOTE and explore cross-validation time series modeled over months or quarters to see if it yields better performance. We would also aim to improve the generalizability of our model by adding a validation dataset and pulling more previous history data. In addition, we would like to train a Neural Network by implementing a multilayer perceptron (MLP) model and experimenting with different NLP network architectures. Finally, we hope to make code/model implementation changes to reduce the runtime for certain operations, such as the main table join.

# COMMAND ----------

# MAGIC %md
# MAGIC ## References
# MAGIC BUREAU OF TRANSPORTATION STATISTICS. (2022). OST_R | BTS | Transtats. OST_R | BTS | Transtats. Retrieved July 31, 2022, from https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
# MAGIC 
# MAGIC Flight delays in numbers - not only painful for passengers. All Things On Time Performance. Retrieved July 22, 2022, from https://www.allthingsontimeperformance.com/flight-delays-in-numbers-not-only-painful-for-passengers
# MAGIC 
# MAGIC Federal Aviation Administration. (2021, June 15). Surface Weather Observation Stations – ASOS/AWOS. Federal Aviation Administration. Retrieved July 31, 2022, from https://www.faa.gov/air_traffic/weather/asos/
# MAGIC 
# MAGIC National Centers for Environmental Information. (2022). Quality Controlled Local Climatological Data (QCLCD) Publication. National Centers for Environmental Information. Retrieved July 31, 2022, from https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00679
# MAGIC 
# MAGIC Partow, A. (2017). The Global Airport Database - By Arash Partow. Arash Partow's Website (www.partow.net). Retrieved July 17, 2022, from https://www.partow.net/miscellaneous/airportdatabase/index.html#Downloads
# MAGIC 
# MAGIC Kaggle. (2020). Cambridge Dictionary | English Dictionary, Translations & Thesaurus. Retrieved August 2, 2022, from https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021?select=US+Holiday+Dates+%282004-2021%29.csv

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Appendix A
# MAGIC 
# MAGIC ### Notebook Links/Summaries
# MAGIC 
# MAGIC [Phase 1 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/3326810404140493/command/28620905305941) \
# MAGIC **Summary**: In Phase 1, the team compiled a report detailing the project abstract, preliminary analysis of data, initial EDA, machine learning algorithms and standard metrics to be used, pipeline steps, and task splitting for future phases of the project. During this stage of the project, we gained a deeper understanding of the data by taking note of the different column values we will be analyzing, along with identifying any null values and other values necessary to be imputed. We used this information to begin thinking about which columns of the tables we would want to be joining for Phase II. For the machine learning algorithms, we decided to use random forest, logistic regression, and gradient boosted tree ensembles for reasons listed in the Machine Learning Algorithms section, and our metrics used will be accuracy, precision, recall, and F-1 score. We have also included a block diagram that lists the projected schedule of tasks for the rest of the project.
# MAGIC   
# MAGIC [Phase 2 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/3326810404140495/command/3326810404140496) \
# MAGIC **Summary**: In Phase 2, the group worked on visualizing more features of all the tables, along with performing the main join across all the tables. We first began by joining the airport and weather station data, and then joined the weather data to the airport/weather station data, and then adding the flight data to the combination of these previously joined data sets. For feature engineering, we standardized the format of timestamps, grouped weather data into hourly stats, converted categorical variables, normalized numerical columns, and imputed null values with 0. We also implemented PCA for dimensionality reduction and plan to combine the resulting principle components with the delay_labels to use for modeling. Finally, we created our baseline logistic regression model by first splitting our data into training/test set, performing feature selection, assembling the features as an input vector to feed into our LM (ridge regression model), and then build the pipeline. The F1_scores of our model are 0.74014 (training set) and 0.74095 (testing set), which are satisfactory as they are generated based on a draft logistic regression model without hyper-tuning and cross-validation on Time Series.
# MAGIC   
# MAGIC [Phase 3 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/3326810404140497/command/28620905306066) \
# MAGIC **Summary**: In Phase 3, the team worked on identifying new features to create a better predictive model, fine-tuning our pipeline using grid earch, implementing two different models with hyper parameter tuning and cross-validation: the XGBoost model and logistic regression model. We also analyzed differences in performance and performed experiments on all data for the new features and experimental settings and reported evaluation metrics over the dataset. One of the main highlights of this phase was using the PageRank analogy and graph theory lessons learned in class and applying them to our model. We used the PageRank method to rank airports because airplane flight paths form a node-link graph with each airport as a node and a flight in between as an edge. We analyzed our results from the two hyperparameter tuned models, including random 3 fold CV, custom fold CV, and XGBoost, and considered which features we want to optimize further for future iterations. Our XGBoost model resulted in an F1 score of 0.929060 and our best hypertuned logistic regression model had an F1 of 0.8676.
# MAGIC   
# MAGIC [Phase 4 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/3326810404140499/command/849916349911001) \
# MAGIC **Summary**: Phase 4 contains the final iteration of feature engineering and model tuning based on feedback from Phase 3. During this phase, the team realized some features should not be included because they contain information not known 2 hours before flight departure. The team also fixed some feature labeling to increase predictability. Since the flight dataset is imbalanced, undersampling was performed to create more balanced data for model training and testing. And finally, additional parameters are experimented to further tune the model. Our final best hypertuned model results are as follows: LR - 0.58475, Random Forest - 0.56793, Tree Ensemble - 0.38410.

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix B
# MAGIC  
# MAGIC ### Tree Ensemble Experiments Results - (Phase 4 Only)
# MAGIC 
# MAGIC Experiments 1-12 were performed in Phase 3 and was done on a dataset containing data leakage.
# MAGIC 
# MAGIC Table of  Parameters
# MAGIC 
# MAGIC | Experiment | Depth | # Trees | Tree Method | Learning Rate | L1    | L2    | Gamma |
# MAGIC |------------|-------|---------|-------------|---------------|-------|-------|-------|
# MAGIC | 13         | 15    | 100     | Approx      | 0.1           | 0.1   | 2     | 0     |
# MAGIC | 14         | 15    | 10      | Approx      | 0.1           | 0.1   | 2     | 0     |
# MAGIC | 15         | 10    | 10      | Approx      | 0.1           | 0.1   | 2     | 0     |
# MAGIC | 16         | 6     | 10      | Approx      | 0.1           | 0.1   | 2     | 0     |
# MAGIC | 17         | 3     | 10      | Approx      | 0.1           | 0.1   | 2     | 0     |
# MAGIC | 18         | 6     | 10      | Approx      | 0.1           | 0.1   | 1E-05 | 0     |
# MAGIC | 19         | 6     | 10      | Approx      | 0.1           | 0.1   | 1E-03 | 0     |
# MAGIC | 20         | 6     | 10      | Approx      | 0.1           | 0.1   | 1E-01 | 0     |
# MAGIC | 21         | 6     | 10      | Approx      | 0.1           | 0.1   | 1     | 0     |
# MAGIC | 22         | 6     | 10      | Approx      | 0.1           | 1E-03 | 1E-05 | 0     |
# MAGIC | 23         | 6     | 50      | Approx      | 0.1           | 0.1   | 1E-05 | 0     |
# MAGIC | 24         | 6     | 10      | Approx      | 0.1           | 0.1   | 1E-05 | 0     |
# MAGIC | 25         | 6     | 10      | Hist        | 0.1           | 0.1   | 1E-05 | 0     |
# MAGIC | 26         | 6     | 10      | Approx      | 1E-03         | 0.1   | 1E-05 | 0     |
# MAGIC | 27         | 6     | 10      | Approx      | 1             | 0.1   | 1E-05 | 0     |
# MAGIC | 28         | 6     | 10      | Approx      | 0.1           | 0.1   | 1E-05 | 1E-03 |
# MAGIC | 29         | 6     | 10      | Approx      | 0.1           | 0.1   | 1E-05 | 1     |
# MAGIC | 30         | 6     | 10      | Approx      | 0.1           | 0.1   | 1E-05 | 5     |
# MAGIC | 31         | 6     | 10      | Approx      | 0.1           | 0.1   | 3     | 0     |
# MAGIC | 32         | 10    | 50      | Approx      | 1             | 0.1   | 2     | 0     |
# MAGIC | 33         | 10    | 50      | Approx      | 1             | 0.1   | 1E-05 | 0     |
# MAGIC | 34         | 10    | 10      | Approx      | 1             | 0.1   | 1E-05 | 0     |
# MAGIC | 35         | 3     | 10      | Approx      | 1             | 0.1   | 1E-05 | 0     |
# MAGIC | 36         | 6     | 10      | Approx      | 1             | 0.1   | 1E-05 | 0     |
# MAGIC | 37         | 6     | 10      | Approx      | 1             | 0.1   | 1E-05 | 5     |
# MAGIC | 38         | 6     | 10      | Approx      | 1             | 0.1   | 1E-05 | 7     |
# MAGIC | 39         | 6     | 10      | Approx      | 1             | 0.1   | 1E-05 | 9     |
# MAGIC | 40         | 6     | 10      | Approx      | 0.1           | 0.1   | 1E-05 | 7     |
# MAGIC | 41         | 6     | 50      | Approx      | 0.1           | 0.1   | 1E-05 | 7     |
# MAGIC | 42         | 6     | 50      | Approx      | 1E-03         | 0.1   | 1E-05 | 7     |
# MAGIC | 43         | 6     | 50      | Approx      | 1             | 0.1   | 1E-05 | 7     |
# MAGIC | 44         | 6     | 50      | Approx      | 1             | 0.1   | 1E-05 | 7     |
# MAGIC | 45         | 6     | 70      | Approx      | 0.1           | 0.1   | 1E-05 | 7     |
# MAGIC | 46         | 6     | 90      | Approx      | 0.1           | 0.1   | 1E-05 | 7     |
# MAGIC 
# MAGIC 
# MAGIC Table of Result Metrics
# MAGIC 
# MAGIC | Experiment | AUC      | RMSE     | Recall            | Precision         | F1                |
# MAGIC |------------|----------|----------|-------------------|-------------------|-------------------|
# MAGIC | 13         | 0.562673 | 0.554452 | 0.518007569101979 | 0.583117834578208 | 0.401849921750139 |
# MAGIC | 14         | 0.560132 | 0.520418 | 0.51581518396325  | 0.583483027600039 | 0.394027582722513 |
# MAGIC | 15         | 0.566745 | 0.521992 | 0.510869317602174 | 0.638836547151706 | 0.365332122537391 |
# MAGIC | 16         | 0.522578 | 0.522578 | 0.508010847527463 | 0.6527387562984   | 0.356161691016823 |
# MAGIC | 17         | 0.507355 | 0.505398 | 0.511826598034331 | 0.613279143555794 | 0.372058587577511 |
# MAGIC | 18         | 0.551806 | 0.522588 | 0.508039570892732 | 0.653093553228085 | 0.356211430253404 |
# MAGIC | 19         | 0.546446 | 0.522588 | 0.508039570892732 | 0.653093553228085 | 0.356211430253404 |
# MAGIC | 20         | 0.548791 | 0.522588 | 0.508039570892732 | 0.653093553228085 | 0.356211430253404 |
# MAGIC | 21         | 0.548791 | 0.522588 | 0.508039570892732 | 0.653093553228085 | 0.356211430253404 |
# MAGIC | 22         | 0.548791 | 0.522588 | 0.508039570892732 | 0.653093553228085 | 0.356211430253404 |
# MAGIC | 23         | 0.574113 | 0.527669 | 0.515936020189554 | 0.635616707714338 | 0.379799582213816 |
# MAGIC | 24         | 0.546543 | 0.522438 | 0.508293128875797 | 0.653350473001046 | 0.356870824652384 |
# MAGIC | 25         | 0.562166 | 0.519264 | 0.508931976137817 | 0.651827050142575 | 0.358708347030614 |
# MAGIC | 26         | 0.538465 | 0.499938 | 0.510125976718227 | 0.642516871650276 | 0.362853812303314 |
# MAGIC | 27         | 0.5645   | 0.601816 | 0.512252000978575 | 0.626686734760637 | 0.370953339106211 |
# MAGIC | 28         | 0.546454 | 0.522588 | 0.508039570892732 | 0.653093553228085 | 0.356211430253404 |
# MAGIC | 29         | 0.522588 | 0.546448 | 0.508039570892732 | 0.653093553228085 | 0.356211430253404 |
# MAGIC | 30         | 0.551808 | 0.522574 | 0.508184673410385 | 0.650080850167671 | 0.356840881768328 |
# MAGIC | 31         | 0.548784 | 0.522577 | 0.50801579983182  | 0.652816644927499 | 0.356168981921513 |
# MAGIC | 32         | 0.512867 | 0.618437 | 0.510364182557786 | 0.528968278131613 | 0.437909266075769 |
# MAGIC | 33         | 0.535799 | 0.632657 | 0.52286478921507  | 0.548827861554589 | 0.487724354910178 |
# MAGIC | 34         | 0.534653 | 0.576747 | 0.512187125791502 | 0.554629684447431 | 0.398836684505476 |
# MAGIC | 35         | 0.53457  | 0.73216  | 0.501556014028888 | 0.678271897790261 | 0.338149897626638 |
# MAGIC | 36         | 0.564494 | 0.601816 | 0.512252000978575 | 0.626686734760637 | 0.370953339106211 |
# MAGIC | 37         | 0.565983 | 0.631743 | 0.510593969479939 | 0.6364955598039   | 0.364849809924082 |
# MAGIC | 38         | 0.567439 | 0.63116  | 0.510670730197468 | 0.637114043400774 | 0.364988898133818 |
# MAGIC | 39         | 0.551841 | 0.615714 | 0.5121301742914   | 0.63562396567842  | 0.369264007213882 |
# MAGIC | 40         | 0.551698 | 0.522629 | 0.508111874536341 | 0.652964553797952 | 0.356415350114638 |
# MAGIC | 41         | 0.579039 | 0.523856 | 0.517043355443722 | 0.629146788496851 | 0.384103396424094 |
# MAGIC | 42         | 0.53847  | 0.499826 | 0.510384982236084 | 0.64269551921093  | 0.363546274791487 |
# MAGIC | 43         | 0.552503 | 0.640637 | 0.511190722154926 | 0.611125982461654 | 0.370526271199179 |
# MAGIC | 44         | 0.565126 | 0.616252 | 0.511024324728539 | 0.612453593692672 | 0.369746107462991 |
# MAGIC | 45         | 0.577227 | 0.527522 | 0.516460469220933 | 0.631741164501957 | 0.381963427574354 |
# MAGIC | 46         | 0.574435 | 0.531857 | 0.516405498642573 | 0.637539018936671 | 0.380740948759118 |
