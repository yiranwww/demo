# the optimized two-pointer approach
class Solution:
    def trap(self, height: List[int]) -> int:
        left, right = 0, len(height)-1
        ans = 0
        left_max, right_max = 0, 0

        while left < right:
            if height[left] < height[right]:
                left_max = max(left_max, height[left])
                ans += left_max - height[left]
                left += 1
            else:
                right_max = max(right_max, height[right])
                ans += right_max - height[right]
                right -=1 
        return ans
    


# the dp approach
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n == 1:
            return 0
        
        left_max = [0] * n
        left_max[0] = height[0]
        right_max = [0] * n
        right_max[-1] = height[-1]
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], height[i])
        
        for i in range(n-2, -1, -1):
            right_max[i] = max(right_max[i+1], height[i])
        
        ans = 0
        for i in range(1, n-1):
            ans += min(left_max[i], right_max[i]) - height[i]
        
        return ans