import numpy as np
import pandas as pd

import pyspark.sql
from pyspark.sql import SparkSession,DataFrame
from pyspark.sql import functions as f
from pyspark.sql import window as win
from pyspark.sql.types import *

spark = SparkSession.builder.appName('test').master('local').getOrCreate()
spark.conf.set('spark.sql.debug.maxToStringFields',1000)

base_df_pd = pd.read_excel(r'/Users/saim/Documents/Work/Spark_PySpark/Datasets/bank.xlsx')

base_df = spark.createDataFrame(base_df_pd)
base_df = base_df.withColumn('targetCol', f.when(base_df['y'] == 'yes', 1).otherwise(0)) \
    .drop(base_df['y'])

mono_bin_list = ['targetCol']
char_bin_list = ['targetCol']

for i, j in base_df.dtypes:
    max_value = base_df.select(i).rdd.max()[0]
    max_value = str(max_value)
    if ((base_df.schema[i].dataType == LongType()) | (base_df.schema[i].dataType == IntegerType())) :
        mono_bin_list.append(i)
    else:
        char_bin_list.append(i)

mono_bin_df = base_df.select(*mono_bin_list)
char_bin_df = base_df.select(*char_bin_list)

mono_bin_df = mono_bin_df.select([f.col(c).cast(IntegerType()) for c in mono_bin_df.columns])
char_bin_df = char_bin_df.select([f.col(c).cast(StringType()) for c in char_bin_df.columns])


def unpivot_data(df:DataFrame) -> DataFrame:
    expr_columns = ', '.join(map(lambda col: '"{col}", {col}'.format(col=col), df.columns))
    expr = "stack({num}, {columns}) as (var_name, values)".format(columns=expr_columns,num=len(df.columns))
    df_stack = df.selectExpr("targetCol",expr).orderBy(f.col('var_name'),f.col('values'))
    return df_stack


char_bin_unpivoted_df = unpivot_data(char_bin_df)

# char_bin_unpivoted_df = char_bin_unpivoted_df.withColumn('MIN_VALUE', char_bin_unpivoted_df['values'])\
#                                              .withColumn('MAX_VALUE',char_bin_unpivoted_df['values'])\
#                                              .withColumn('COUNT', char_bin_unpivoted_df.groupby(char_bin_unpivoted_df['values']).count()['count'])


#['targetCol', 'var_name', 'values']

Count_df = char_bin_unpivoted_df.groupby(char_bin_unpivoted_df['values']).agg(f.count(char_bin_unpivoted_df['values']).alias('count'))
char_bin_unpivoted_df = char_bin_unpivoted_df.withColumn('MIN_VALUE', char_bin_unpivoted_df['values'])\
                                             .withColumn('MAX_VALUE',char_bin_unpivoted_df['values'])\



#char_bin_unpivoted_df = char_bin_unpivoted_df.withColumn('Count_value', char_bin_unpivoted_df.groupby(char_bin_unpivoted_df['var_name']).count())
cnt_df = char_bin_unpivoted_df.groupby(char_bin_unpivoted_df['values'].alias('cnt_var'),char_bin_unpivoted_df['targetCol']).count()
char_bin_unpivoted_df = char_bin_unpivoted_df.join(cnt_df,char_bin_unpivoted_df['values'] == cnt_df['cnt_var'], 'left')
char_bin_unpivoted_df = char_bin_unpivoted_df.drop('cnt_var').dropDuplicates()

char_bin_unpivoted_df.show()
# justmiss = df_stack.where(f.col('target').isNull())

# notmiss = df_stack.where(f.col('target').isNotNull())
#
# ntile_df = notmiss.withColumn('Bucket',f.ntile(20)
#                               .over(win.Window
#                                     .partitionBy(f.col('var_name'),f.col('target'))
#                                     .orderBy(f.col('var_name'),f.col('target'))))
#
# ntile_df_groupby = ntile_df.groupby('bucket').count()
# ntile_df_groupby.show()