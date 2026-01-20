1. If running time is too long, try to transfer Pandas into Numpy

2.  Jaccard Similarity:
   ```
import numpy as np
g = events_df.groupby("company_id")
a = g.get_group(1)['user_id']
b = g.get_group(2)['user_id']
test = len(np.intersect1d(a.values, b.values))/ len(np.union1d(a.values, b.values))


print(test)
```
