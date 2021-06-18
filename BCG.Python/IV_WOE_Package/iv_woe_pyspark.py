import datetime

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql import functions as f
import configparser

#Configuration builder
configParser = configparser.RawConfigParser()
configFilePath = r'config.conf'
configParser.read(configFilePath)

#Configuration values
path = configParser.get('input_file_path', 'path')
MAX_BIN = configParser.get('Bins', 'MAX_BIN')

spark = SparkSession.builder.appName('IV_WOE').master('local').getOrCreate()
spark.conf.set("spark.sql.debug.maxToStringFields", 1000)



base_df_pd = pd.read_excel(path)

base_df = spark.createDataFrame(base_df_pd)
base_df = base_df.withColumn('targetCol', f.when(base_df['y'] == 'yes', 1).otherwise(0)).drop(base_df['y'])


def unpivot_data(df: DataFrame,target) -> DataFrame:
    expr_columns = ', '.join(map(lambda col: '"{col}", {col}'.format(col=col), df.columns))
    expr = "stack({num}, {columns}) as (var_name, values)".format(columns=expr_columns, num=len(df.columns))
    df_stack = df.selectExpr(target, expr).orderBy(f.col('var_name'), f.col('values'))
    return df_stack

def mono_bin(df:DataFrame,target):
    EVENT_sum, nonEVENT_sum = get_event_nonevent_sum(df, target)
    mono_bin_df = unpivot_data(df,target)
    mono_bin_df = mono_bin_df.where(f.col('var_name') != target)
    mono_bin_df = mono_bin_df.withColumn('bucket', f.ntile(MAX_BIN).over(
        Window.partitionBy(f.col('var_name')).orderBy(f.col('values'))))
    mono_bin_df_grouped = mono_bin_df.groupby(f.col('var_name'), f.col('bucket'))

    # MAX, MIN and COUNT
    mono_cal_df = mono_bin_df_grouped.agg(f.min(f.col('values')).alias('MIN_VALUE')
                                          , f.max(f.col('values')).alias('MAX_VALUE')
                                          , f.count(f.col(target)).alias('COUNT_VALUE')
                                          , f.sum(f.col(target)).alias('EVENT')
                                          ,
                                          (f.count(f.col(target)) - f.sum(f.col(target))).alias('NON_EVENT'))

    mono_cal_df = mono_cal_df.withColumn('EVENT_RATE', (f.col('EVENT') / f.col('COUNT_VALUE')).cast(FloatType())) \
        .withColumn('NONEVENT_RATE', (f.col('NON_EVENT') / f.col('COUNT_VALUE')).cast(FloatType())) \
        .withColumn('DIST_EVENT', (f.col('EVENT') / EVENT_sum).cast(FloatType())) \
        .withColumn('DIST_NON_EVENT', (f.col('NON_EVENT') / nonEVENT_sum).cast(FloatType()))

    mono_cal_df = mono_cal_df.withColumn('WOE',
                                         (f.log(f.col('DIST_EVENT') / f.col('DIST_NON_EVENT'))).cast(FloatType()))
    mono_cal_df = mono_cal_df.withColumn('IV', ((f.col('DIST_EVENT') - f.col('DIST_NON_EVENT')) * f.col('WOE')).cast(
        FloatType())).drop('bucket')

    return mono_cal_df


def char_bin(df: DataFrame, target):

    EVENT_sum,nonEVENT_sum = get_event_nonevent_sum(df,target)

    char_bin_unstacked_df = unpivot_data(df,target)
    char_bin_unstacked_df = char_bin_unstacked_df.withColumn('MIN_VALUE', char_bin_unstacked_df['values']).withColumn \
        ('MAX_VALUE', char_bin_unstacked_df['values'])

    Count_df = char_bin_unstacked_df.groupby(char_bin_unstacked_df['values']).agg \
        (f.count(char_bin_unstacked_df['values']).alias('count'))

    EVENT_df = char_bin_unstacked_df.where(char_bin_unstacked_df[target] == 1).groupBy \
        (char_bin_unstacked_df['values']).agg(f.sum(char_bin_unstacked_df[target]).alias('EVENT'))

    NONEVENT_DF = Count_df.join(EVENT_df, ['values']).withColumn('NONEVENT', (Count_df['count'] - EVENT_df['EVENT']))

    NONEVENT_DF = NONEVENT_DF.withColumn('EVENT_RATE', (NONEVENT_DF['EVENT'] / NONEVENT_DF['count']).cast(FloatType())) \
        .withColumn('NONEVENT_RATE',
                    (NONEVENT_DF['NONEVENT'] / NONEVENT_DF['count']).cast(FloatType())).withColumnRenamed \
        ('values', 'values_refined')

    char_bin_unstacked_df = char_bin_unstacked_df.join(NONEVENT_DF,
                                                       char_bin_unstacked_df['values'] == NONEVENT_DF['values_refined'],
                                                       'left')

    char_bin_unstacked_df = char_bin_unstacked_df.select(char_bin_unstacked_df[target].cast(IntegerType()),
                                                         'var_name'
                                                         , 'values', 'MIN_VALUE', 'MAX_VALUE', 'count', 'EVENT',
                                                         'NONEVENT'
                                                         , 'EVENT_RATE', 'NONEVENT_RATE')

    char_bin_unstacked_df = char_bin_unstacked_df.drop(target)

    char_bin_unstacked_df = char_bin_unstacked_df.withColumn('DIST_EVENT',
                                                             (char_bin_unstacked_df['EVENT'] / EVENT_sum).cast(
                                                                 FloatType())) \
        .withColumn('DIST_NONEVENT', (char_bin_unstacked_df['NONEVENT'] / nonEVENT_sum).cast(FloatType()))

    char_bin_unstacked_df = char_bin_unstacked_df.distinct()

    char_bin_unstacked_df = char_bin_unstacked_df.withColumn('WOE', f.log(
        char_bin_unstacked_df['DIST_EVENT'] / char_bin_unstacked_df['DIST_NONEVENT']).cast(FloatType()))

    char_bin_unstacked_df = char_bin_unstacked_df.withColumn('IV',
                                                             ((char_bin_unstacked_df['DIST_EVENT'] -
                                                               char_bin_unstacked_df[
                                                                   'DIST_NONEVENT']) * char_bin_unstacked_df['WOE']))

    char_bin_unstacked_df = char_bin_unstacked_df.where(char_bin_unstacked_df['var_name'] != target).drop('values')

    return char_bin_unstacked_df

def get_event_nonevent_sum(df:DataFrame, target):
    sum_EVENT_nonEVENT = base_df.select(f.col(target)).groupBy(base_df[target]).count().withColumnRenamed \
        (target, 'targetValues').withColumnRenamed('count', 'sum_value')
    EVENT_sum = sum_EVENT_nonEVENT.where(sum_EVENT_nonEVENT['targetValues'] == 1).select('sum_value').first()[0]
    nonEVENT_sum = sum_EVENT_nonEVENT.where(sum_EVENT_nonEVENT['targetValues'] == 0).select('sum_value').first()[0]

    return EVENT_sum,nonEVENT_sum




def get_iv_woe(df:DataFrame,target):
    base_df = df
    #target = 'targetCol'
    mono_bin_list = [target]
    char_bin_list = [target]

    for i, j in base_df.dtypes:
        if (base_df.schema[i].dataType == LongType()) | (base_df.schema[i].dataType == IntegerType()):
            if i != target:
                mono_bin_list.append(i)
            mono_bin_df = base_df.select(*mono_bin_list)
            mono_bin_df = mono_bin_df.select([f.col(c).cast(IntegerType()) for c in mono_bin_df.columns])
            mono_bin_iv_woe = mono_bin(mono_bin_df,target)
        else:
            char_bin_list.append(i)
            char_bin_df = base_df.select(*char_bin_list)
            char_bin_df = char_bin_df.select([f.col(c).cast(StringType()) for c in char_bin_df.columns])
            char_bin_iv_woe = char_bin(char_bin_df,target)

    IV_WOE_df = char_bin_iv_woe.union(mono_bin_iv_woe).orderBy(f.col('var_name'), f.col('MIN_VALUE'))
    return IV_WOE_df


#Unit Testing
start_time = datetime.datetime.now()
test_df = get_iv_woe(base_df,'targetCol')
end_time = datetime.datetime.now()
test_df.show()
print(start_time)
print(end_time)
