class Solution:
    def firstUniqChar(self, s: str) -> int:
        helper = collections.Counter(s)
        for i, cur in enumerate(s):
            if helper[cur] == 1:
                return i
        return -1