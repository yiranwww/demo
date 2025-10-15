# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        res = []

        def dfs(node, path, res):
            if not node:
                return 
            path += str(node.val)
            if not node.left and not node.right:
                res.append(path)
            else:
                dfs(node.left, path + '->', res)
                dfs(node.right, path + '->', res)
        
        dfs(root, '', res)
        return res
    

# another method with BP
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        def dfs(node, path, res):
            if not node:
                return 
            path.append(str(node.val))
            if not node.left and not node.right:
                res.append('->'.join(path))
            else:
                dfs(node.left, path, res)
                dfs(node.right, path, res)
            path.pop()
        
        
        res =[]
        dfs(root, [], res)
        return res