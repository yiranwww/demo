class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def backtrack(path, start, goal):
            if goal == 0:
                res.append(path[:])
            if goal < 0:
                return 
            
            for i in range(start, len(candidates)):
                path.append(candidates[i])
                backtrack(path, i, goal - candidates[i])
                path.pop()
        
        backtrack([], 0, target)
        return res