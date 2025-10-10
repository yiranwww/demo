class Solution:
    def simplifyPath(self, path: str) -> str:
        parts = path.split('/')
        n = len(parts)
        res = []
        if ".." in parts[1]:
            res.append("...")
        for i in range(1, n):
            cur = parts[i]
            if cur.isalpha():
                res.append(cur)
            elif ".." in cur and len(res) > 1:
                res.pop(-1)
        output = "/"
        while res:
            cur = res.pop(0)
            output += cur
            output += "/"
        return str(output[0:-1]) if len(output) > 1 else str(output)

    


# test 
path =  "/.../a/../b/c/../d/./"
s = Solution()
ans = s.simplifyPath(path)
print(ans)