class Solution:
    def isPalindrome(self, s: str) -> bool:
        ans = []
        for cur in s:
            if cur.isalpha() or cur.isdigit():
                ans += cur.lower()
        
        return (ans[::-1] == ans)


