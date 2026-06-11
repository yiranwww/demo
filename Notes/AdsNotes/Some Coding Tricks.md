1. If running time is too long, try to transfer *Pandas* into *Numpy*

2.  Jaccard Similarity:
   ```
import numpy as np
g = events_df.groupby("company_id")
a = g.get_group(1)['user_id']
b = g.get_group(2)['user_id']
test = len(np.intersect1d(a.values, b.values))/ len(np.union1d(a.values, b.values))


print(test)
```
*intersect1d* Return the sorted, unique values that are in both of the input arrays.
即返回数组中的原始值。如果需要计算ratio或个数需要用*len（）*。
*union1d* 返回两个数组的所有去重值。类似Sql里面的Join。
