# 53. Maximum Subarray
# Given an integer array nums, find the subarray with the largest sum, and return its sum.

 

# Example 1:

# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6
# Explanation: The subarray [4,-1,2,1] has the largest sum 6.
# Example 2:

# Input: nums = [1]
# Output: 1
# Explanation: The subarray [1] has the largest sum 1.
# Example 3:

# Input: nums = [5,4,-1,7,8]
# Output: 23
# Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
 

# Constraints:

# 1 <= nums.length <= 105
# -104 <= nums[i] <= 104
 

# Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

class Solution:
    def maxSubArray(self, nums):
        n = len(nums)
        if n <= 1:
            return nums[0]

        left_sum = [0] * n
        right_sum = [0] * n

        for i in range(1, n):
            left_sum[i] = max(0, left_sum[i-1] + nums[i-1])
        
        for i in range(n-2, -1, -1):
            right_sum[i] = max(0, right_sum[i+1] + nums[i+1])
        
        res = float('-inf')
        for i in range(n):
            res = max(res, nums[i] + left_sum[i] + right_sum[i])
        
        return res

# test 
nums = [5,4,-1,7,8]
s = Solution()
ans = s.maxSubArray(nums)
print(ans)