class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        cur_k = k - 1
        if cur_k == 0:
            return True
        for i in range(k+1, len(nums)):
            if nums[i] > nums[i-1] and nums[i-k] > nums[i-k-1]:
                cur_k -=1
            else:
                cur_k = k - 1
            if cur_k == 0:
                return True
        return False