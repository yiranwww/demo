# We define a harmonious array as an array where the difference between its maximum value and its minimum value is exactly 1.

# Given an integer array nums, return the length of its longest harmonious subsequence among all its possible subsequences.

 

# Example 1:

# Input: nums = [1,3,2,2,5,2,3,7]

# Output: 5

# Explanation:

# The longest harmonious subsequence is [3,2,2,2,3].

# Example 2:

# Input: nums = [1,2,3,4]

# Output: 2

# Explanation:

# The longest harmonious subsequences are [1,2], [2,3], and [3,4], all of which have a length of 2.

# Example 3:

# Input: nums = [1,1,1,1]

# Output: 0

# Explanation:

# No harmonic subsequence exists.

 

# Constraints:

# 1 <= nums.length <= 2 * 104
# -109 <= nums[i] <= 109

# Method: 
# sort the array, and use two pointers to find the longest harmonious subsequence.

from typing import List
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        nums.sort()
        j = 0
        maxLength = 0
        for i in range(len(nums)):
            while nums[i] - nums[j] > 1:
                j += 1
            if nums[i] -nums[j] == 1:
                maxLength = max(maxLength, i - j + 1)
        return maxLength
    
# Example usage:
solution = Solution()
nums = [1,3,2,2,5,2,3,7]
ans = solution.findLHS(nums)
print(ans)  # Output: 5
