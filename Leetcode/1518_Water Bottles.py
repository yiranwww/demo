class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        res = 0
        we_have = numBottles
        kong = 0
        while (we_have + kong) >= numExchange:
            res += we_have
            new = (we_have + kong) // numExchange
            kong = (we_have + kong) % numExchange
            we_have = new
        return res + we_have