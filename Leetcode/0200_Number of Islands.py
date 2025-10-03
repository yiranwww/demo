class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        res = 0
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def dfs(i,j):
            grid[i][j] = "0"
            for dx, dy in directions:
                newx = i + dx
                newy = j + dy
                if (0 <= newx < m and 0 <= newy < n and grid[newx][newy] == "1"):
                    dfs(newx, newy)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    dfs(i, j)
                    res += 1
        return res