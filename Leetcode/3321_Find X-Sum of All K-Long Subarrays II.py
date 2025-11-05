import collections
from typing import List
class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        
        def cal_sub_sum(num, top_k):
            helper = collections.Counter(num)
            sort_helper = sorted(helper.items(), key=lambda x: (-x[1], -x[0]))

            res = 0
            for i in range(top_k):
                val, freq = sort_helper[i]
                res += val * freq

            # for val, freq in sort_helper: 
            #     while top_k:
            #         res += val * freq
            #         break
            #     top_k -=1
            return res
        
        n = len(nums)
        ans = []

        if n <= k:
            return sum(nums)
        for i in range(0, n-k+1):
            cur_subarry = nums[i:i + k]
            cur_sum = cal_sub_sum(cur_subarry, x)
            ans.append(cur_sum)
        return ans



## test case
S = Solution()
nums = [1,1,2,2,3,4,2,3]
k = 6
x = 2
ans = S.findXSum(nums, k, x)
print(ans)