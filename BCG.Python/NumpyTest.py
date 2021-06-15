import numpy as np
import pandas as pd

arr = np.array([1,2,3,4,5])
labels = ['a','b','c','d','e']

pd.Series(data=arr ,index=labels)


data1=['Sai',25,'Analyst','BCG','BLR','1994-04-04']
index2=[1,2,3,4,5,6]
df = pd.DataFrame(data1,index2,columns=['Details'])
#print(df)

list1 = 'values' * 4

print(list1)

# import pyspark.sql.functions as f
#
#
# expr_columns = ', '.join(map(lambda col: '"{col}", {col}'.format(col=col), df.columns))
# expr = "stack(2, {columns}) as (var_name, values)".format(columns=expr_columns)
#
# df_stack = df.selectExpr(expr)
# df_final = df_stack.groupBy("var_name").agg(f.collect_list(f.col("values")))