import heapq
class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        heapq.heapify(nums)
        arr = []
        alice = []
        bob = []
        while nums:
            alice = heapq.heappop(nums)
            bob = heapq.heappop(nums)
            arr.append(bob)
            arr.append(alice)
        return arr
    

# more straightforward solution
class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        n = len(nums)
        arr = []
        for i in range(0, n, 2):
            arr.append(nums[i+1])
            arr.append(nums[i])
        return arr