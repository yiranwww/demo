import heapq
class Solution:
    def lastStoneWeight(self, stones):
        heap = [-num for num in stones]
        heapq.heapify(heap)

        while len(heap) > 1:
            x = -heapq.heappop(heap)
            y = -heapq.heappop(heap)
            if x != y:
                heapq.heappush(heap, -(x-y))
        return -heap[0] if heap else 0
    

# test 
stones = [2,7,4,1,8,1]
# Output: 1
s = Solution()
ans = s.lastStoneWeight(stones)
print(ans)