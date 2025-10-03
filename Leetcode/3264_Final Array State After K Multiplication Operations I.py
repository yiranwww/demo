import heapq
class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        if multiplier == 1:
            return nums
            
        heap = [(num, idx) for idx, num in enumerate(nums)]
        heapq.heapify(heap)
        
        while k > 0:
            c_num, c_idx = heapq.heappop(heap)
            c_num *= multiplier
            heapq.heappush(heap, (c_num, c_idx))
            k -=1
        
        while heap:
            num, idx = heapq.heappop(heap)
            nums[idx] = num
        return nums