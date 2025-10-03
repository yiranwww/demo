class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        area = 0

        def dfs(i, j):
            if not (0 <= i < m and 0 <= j < n) or grid[i][j] == 0:
                return 0
            grid[i][j] = 0
            area = 1
            for dx, dy in direction:
                newx = i + dx
                newy = j + dy
                area += dfs(newx, newy)
            return area


        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    area = dfs(i, j)
                    res = max(res, area)
        return res
    

### another solution
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def bfs(i, j):
            q = deque([(i, j)])
            grid[i][j] = 0
            area = 1
            while q:
                x, y = q.popleft()
                for dx, dy in directions:
                    newx = x + dx
                    newy = y + dy
                    if (0 <= newx < m and 0 <= newy < n and grid[newx][newy] == 1):
                        area += 1
                        grid[newx][newy] = 0
                        q.append((newx, newy))
            return area

        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    res = max(res, bfs(i, j))
        return res