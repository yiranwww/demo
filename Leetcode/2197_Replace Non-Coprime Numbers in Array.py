# You are given an array of integers nums. Perform the following steps:

# Find any two adjacent numbers in nums that are non-coprime.
# If no such numbers are found, stop the process.
# Otherwise, delete the two numbers and replace them with their LCM (Least Common Multiple).
# Repeat this process as long as you keep finding two adjacent non-coprime numbers.
# Return the final modified array. It can be shown that replacing adjacent non-coprime numbers in any arbitrary order will lead to the same result.

# The test cases are generated such that the values in the final array are less than or equal to 108.

# Two values x and y are non-coprime if GCD(x, y) > 1 where GCD(x, y) is the Greatest Common Divisor of x and y.

 

# Example 1:

# Input: nums = [6,4,3,2,7,6,2]
# Output: [12,7,6]
# Explanation: 
# - (6, 4) are non-coprime with LCM(6, 4) = 12. Now, nums = [12,3,2,7,6,2].
# - (12, 3) are non-coprime with LCM(12, 3) = 12. Now, nums = [12,2,7,6,2].
# - (12, 2) are non-coprime with LCM(12, 2) = 12. Now, nums = [12,7,6,2].
# - (6, 2) are non-coprime with LCM(6, 2) = 6. Now, nums = [12,7,6].
# There are no more adjacent non-coprime numbers in nums.
# Thus, the final modified array is [12,7,6].
# Note that there are other ways to obtain the same resultant array.
# Example 2:

# Input: nums = [2,2,1,1,3,3,3]
# Output: [2,1,1,3]
# Explanation: 
# - (3, 3) are non-coprime with LCM(3, 3) = 3. Now, nums = [2,2,1,1,3,3].
# - (3, 3) are non-coprime with LCM(3, 3) = 3. Now, nums = [2,2,1,1,3].
# - (2, 2) are non-coprime with LCM(2, 2) = 2. Now, nums = [2,1,1,3].
# There are no more adjacent non-coprime numbers in nums.
# Thus, the final modified array is [2,1,1,3].
# Note that there are other ways to obtain the same resultant array.
 

# Constraints:

# 1 <= nums.length <= 105
# 1 <= nums[i] <= 105
# The test cases are generated such that the values in the final array are less than or equal to 108.

# 思路： 用stack
# 每加入一个新元素（cur)，考虑与前一个元素是否互质。

import math
class Solution:
    def replaceNonCoprimes(self, nums):
        stack = []
        n = len(nums)

        def is_coprime(num1, num2):
            return math.gcd(num1, num2) == 1

        def get_lcm(num1, num2):
            return math.lcm(num1, num2)
        

        for i in range(n):
            stack.append(nums[i])
            # check current and previous
            while len(stack) >= 2:
                if not is_coprime(stack[-1], stack[-2]):
                    second = stack.pop()
                    first =stack.pop()
                    stack.append(get_lcm(first, second))
                else:
                    break
        return stack
    

# test 
nums = [287,41,49,287,899,23,23,20677,5,825]
s = Solution()
ans = s.replaceNonCoprimes(nums)
print(ans)