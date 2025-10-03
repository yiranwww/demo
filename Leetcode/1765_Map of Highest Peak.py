class Solution:
    def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
        m, n = len(isWater), len(isWater[0])
        direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        queue = deque()
        res = [[-1] * n for _ in range(m)]

        visited = set()
        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    res[i][j] = 0
                    queue.append((i, j))
        
        while queue:
            r, c = queue.popleft()
            h = res[r][c]
            for dx, dy in direction:
                newx = r + dx
                newy = c + dy
                if (newx < 0 or newx >= m or newy < 0 or newy >= n or res[newx][newy] != -1):
                    continue
                queue.append((newx, newy))
                res[newx][newy] = h + 1
        return res
