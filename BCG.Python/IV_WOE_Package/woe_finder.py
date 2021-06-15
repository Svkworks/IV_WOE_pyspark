import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import traceback,re
from pyspark.sql import functions as f

# import packages
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

spark = SparkSession.builder.appName('test').master('local').getOrCreate()

base_df_pd = pd.read_excel(r'/Users/saim/Documents/Work/Spark_PySpark/Datasets/bank.xlsx')

max_bin = 20
force_bin = 3

# define a binning function
def mono_bin(Y, X, n=max_bin):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1] - (bins[1] / 2)
        d1 = pd.DataFrame(
            {"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True)})
        d2 = d1.groupby('Bucket', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop=True)

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()

    return (d3)


base_df = spark.createDataFrame(base_df_pd)
base_df = base_df.withColumn('target', f.when(base_df['y'] == 'yes', 1).otherwise(0)) \
   .drop(base_df['y'])

df1 = base_df_pd
# Index(['age', 'job', 'marital', 'education', 'default', 'balance',
#      'housing','loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays','previous', 'poutcome', 'y'])

x = df1.dtypes.index
count = -1
n = max_bin

for i in x:
    if np.issubdtype(df1[i], np.number) and len(pd.Series.unique(df1[i])) > 2:
                #conv = mono_bin('target', df1[i])
                df1 = pd.DataFrame({"X": 'target', "Y": df1[i]})
                justmiss = df1[['X', 'Y']][df1.X.isnull()]
                notmiss = df1[['X', 'Y']][df1.X.notnull()]
                r = 0
                while np.abs(r) < 1:
                    try:
                        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
                        d2 = d1.groupby('Bucket', as_index=True)
                        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
                        n = n - 1
                    except Exception as e:
                        n = n - 1

                print('d1 DataFrame---------------------------------------')
                print(d1)
                print('d2 DataFrame---------------------------------------')
                print(d2)
                print('stats----------------------------------------------')
                print(stats.spearmanr(d2.mean().X, d2.mean().Y))
                #conv["VAR_NAME"] = i
                #count = count + 1
