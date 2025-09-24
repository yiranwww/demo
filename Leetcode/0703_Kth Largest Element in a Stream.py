import heapq
class KthLargest:

    def __init__(self, k: int, nums):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)
        
        

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > k:
            heapq.heappop(self.heap)
        return self.heap[0]
   
        
        


# # Your KthLargest object will be instantiated and called as such:
k = 3
nums = [4, 5, 8, 2]
obj = KthLargest(k, nums)
param_1 = obj.add(3)
param_2 = obj.add(5)
param_3 = obj.add(10)
print(param_1,param_2,param_3)