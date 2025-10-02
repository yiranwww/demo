class Solution:
    def maxBottlesDrunk(self, numBottles: int, numExchange: int) -> int:
        res = 0
        full_b = numBottles
        empty_b = 0
        while (full_b + empty_b) >= numExchange:
            res += full_b
            empty_b += full_b - numExchange
            full_b = 1
            numExchange += 1
        return res + full_b
