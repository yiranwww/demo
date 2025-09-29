import heapq
class Solution:
    def fillCups(self, amount: List[int]) -> int:
        heap = [-x for x in amount]
        heapq.heapify(heap)
        res = 0

        # check the largest
        while -heap[0] > 0:
            first = -heapq.heappop(heap)
            second = -heapq.heappop(heap)
            if first > 0:
                first -=1
            if second > 0:
                second -= 1
            heapq.heappush(heap, -first)
            heapq.heappush(heap, -second)
            res += 1
        return res