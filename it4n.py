
# coding: utf-8

# # SETUP & IMPORT EVERYTHING

# In[2]:


# Must be included at the beginning of each new notebook. Remember to change the app name.
import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('it4').getOrCreate()


# In[3]:


from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler,VectorIndexer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf,when,isnan,count,isnull


# # READ

# In[4]:


df = spark.read.csv('aws-instance/Datasets/dtd.csv', header=True, inferSchema=True)


# # TRANSFORM ATTRITION

# In[97]:


def get_attrition(attrition):
  if attrition=="Yes":
    attrition=1
  else:
    attrition=0
  return attrition

udf_attrition = udf(lambda attrition : get_attrition(attrition),IntegerType())
master_df=df.withColumn("attrition_class", udf_attrition(df.Attrition)).drop("Attrition")
#master_df.show()
#master_df.groupby("attrition_class").count().show()


# In[27]:


df


# In[28]:


master_df


# # INDEXERS & ENCODERS → DF

# In[98]:


categoricalCols=["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus",
        "OverTime"]

continuousCols=["Age",'DailyRate','DistanceFromHome',
'Education','EmployeeCount',\

'EnvironmentSatisfaction','HourlyRate',\

'JobInvolvement','JobLevel','JobSatisfaction',\

'MonthlyIncome','MonthlyRate','NumCompaniesWorked',\

'PercentSalaryHike','PerformanceRating','RelationshipSatisfaction',\

'StandardHours','StockOptionLevel','TotalWorkingYears',\

'TrainingTimesLastYear','WorkLifeBalance',\

'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']


# In[99]:


indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
         for c in categoricalCols ]

encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
         outputCol="{0}_encoded".format(indexer.getOutputCol()))
         for indexer in indexers ]


# In[100]:


assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders] + continuousCols, outputCol="features")


# In[101]:


pipeline = Pipeline(stages=indexers + encoders + [assembler])


# In[104]:


model=pipeline.fit(master_df)


# In[103]:


master_df = model.transform(master_df)


# In[19]:


master_df.columns


# In[25]:


master_df.select('features').collect()


# # SPLIT

# In[105]:


splits = master_df.randomSplit([0.9, 0.1])
train_df = splits[0]
test_df = splits[1]


# In[61]:


train_df.columns


# # PCA

# In[53]:


from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors


pca = PCA(k=5, inputCol="features", outputCol="pcaFeatures")
modelpca = pca.fit(master_df)


# In[54]:


pcaresult = modelpca.transform(master_df).select("pcaFeatures","attrition_class")
pcaresult.show(truncate=False)


# # SPLIT PCA

# In[80]:


splitspca = pcaresult.randomSplit([0.9, 0.1])
pcatrain_df = splitspca[0]
pcatest_df = splitspca[1]


# In[81]:


print((pcatrain_df.count(), len(pcatrain_df.columns)))


# In[82]:


print((pcatest_df.count(), len(pcatest_df.columns)))


# # LR

# In[83]:


from pyspark.ml.classification import LogisticRegression

logr = LogisticRegression(maxIter = 1,featuresCol = 'pcaFeatures', labelCol="attrition_class")

#lr = LinearRegression()
logr_model = logr.fit(pcatrain_df)
# Make predictions.
lgpredictions = logr_model.transform(pcatest_df)
lgpredictions_train = logr_model.transform(pcatrain_df)


# In[86]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="attrition_class", predictionCol="prediction", metricName="accuracy")
lgaccuracy = evaluator.evaluate(lgpredictions)
lgaccuracy_train = evaluator.evaluate(lgpredictions_train)
print("Test Accuracy = %g" % (lgaccuracy))
print("Train Accuracy = %g" % (lgaccuracy_train))
predictions_and_labels = logr_model.evaluate(pcatest_df)
evaluatorroc = BinaryClassificationEvaluator(labelCol="attrition_class")
my_final_roc = evaluatorroc.evaluate(predictions_and_labels.predictions)
print("AUC Score =",my_final_roc)


# In[85]:


#ROC
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(logr_model.summary.roc.select('FPR').collect(),
         logr_model.summary.roc.select('TPR').collect())
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


# In[115]:


lgpredictions.select("pcaFeatures","attrition_class","prediction").show(1000)


# In[58]:


from pyspark.ml.classification import LogisticRegression

maxIters = [1,10,100,200]

for i in maxIters:
    logr = LogisticRegression(maxIter = i, featuresCol = 'pcaFeatures', labelCol="attrition_class")
    logr_model1 = logr.fit(pcatrain_df)
    lgpredictions = logr_model1.transform(pcatest_df)
    lgpredictions_train = logr_model1.transform(pcatrain_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="attrition_class", predictionCol="prediction", metricName="accuracy")
    lgaccuracy = evaluator.evaluate(lgpredictions)
    lgaccuracy_train = evaluator.evaluate(lgpredictions_train)
    print(i,":Train Accuracy = %g" % (lgaccuracy_train))
    evaluatorroc = BinaryClassificationEvaluator(labelCol="attrition_class")
    my_final_roc = evaluatorroc.evaluate(predictions_and_labels.predictions)
    print(i,":AUC Score=",my_final_roc)


# # EVALUATION

# In[62]:


#ROC
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(logr_model.summary.roc.select('FPR').collect(),
         logr_model.summary.roc.select('TPR').collect())
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


# In[26]:


# And evaluating it against the test data.
predictions_and_labels = logr_model.evaluate(pcatest_df)

predictions_and_labels.predictions.show()


# In[70]:


#AUC Score
# Let's import two evalulation metrics. 
# Remember, binary is for predictions like true and false (0 and 1), 
# While multi-class is for multiple classification classes.
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

# According to this evaluation metric, the area under the curve is 1.0. A perfect fit? Is that realistic?  
evaluator = BinaryClassificationEvaluator(labelCol="attrition_class")
my_final_roc = evaluator.evaluate(predictions_and_labels.predictions)
my_final_roc


# # Drop rows with missing values

# In[90]:


df1 = spark.read.csv('aws-instance/Datasets/dtd.csv', header=True, inferSchema=True)


# In[19]:


df1.show(4)


# In[20]:


df1.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df1.columns]).show()


# In[91]:


df2 = spark.read.csv('aws-instance/Datasets/dtd2.csv', header=True, inferSchema=True)


# In[21]:


import pyspark.sql.functions as f
from functools import reduce
df1.where(reduce(lambda x, y: x | y, (f.col(x).isNull() for x in df1.columns))).count()


# In[92]:


print((df1.count(), len(df1.columns)))


# In[93]:


# Using this syntax, we can drop any row with missing data. Three rows are dropped.
df1=df1.na.drop()


# In[94]:


print((df1.count(), len(df1.columns)))


# In[95]:


print((df2.count(), len(df2.columns)))


# # Join

# In[96]:


df = df1.union(df2)


# In[10]:


df.collect()


# In[11]:


print((df.count(), len(df.columns)))


# In[34]:


print((master_df.count(), len(master_df.columns)))

