class Solution:
    def isPalindrome(self, s: str) -> bool:
        string = ""
        for cur in s:
            if cur.isalpha() or cur.isdigit():
                string += cur.lower()
        
        return string == string[::-1]

S = Solution()
s = "0P"
ans = S.isPalindrome(s)
print(ans)