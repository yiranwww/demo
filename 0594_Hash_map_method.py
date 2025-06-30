from typing import List
import collections
# Method: Use a hash map to count the frequency of each number, then check for pairs that differ by 1.

class Solution:
    def findLHS(self, nums: List[int]) -> int:
        freq_map = collections.Counter(nums)
        max_length = 0
        for num, freq in freq_map.items():
            if num + 1 in freq_map:
                cur_length = freq_map[num] + freq_map[num+1]
                max_length = max(max_length, cur_length)
        return max_length
    
solution = Solution()
nums = [1,3,2,2,5,2,3,7]
ans = solution.findLHS(nums)
print(ans)  # Output: 5