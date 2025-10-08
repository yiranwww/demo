class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        potions.sort()
        n = len(spells)
        m = len(potions)
        res = [0] * n

        def helper(cur_s, success):
            left = 0
            right = m -1
            while left <= right:
                mid = (left + right) // 2
                if potions[mid] * cur_s < success:
                    left = mid +1
                else:
                    right = mid -1
            return m - left 

        for i in range(n):
            cur_s = spells[i]
            cur_cnt = helper(cur_s,success)
            res[i] = cur_cnt
        
        return res