# Title: All Subsequences of a String
# Difficulty: Medium
# Tag: Backtracking, String, Combinations

"""
Given a string s, return **all possible non-empty subsequences** of s, sorted in **alphabetical order**.  
A subsequence is obtained by deleting zero or more characters from the string while maintaining the original order.

Example 1:
Input: s = "xyz"
Output: ["x", "xy", "xyz", "xz", "y", "yz", "z"]

Example 2:
Input: s = "ba"
Output: ["a", "b", "ba"]

Constraints:
- 1 <= s.length <= 16
- s consists of lowercase English letters.

Follow-up:
- Can you implement it using backtracking?
- Analyze the time and space complexity of your solution.
"""

from typing import List

class Solution:
    def allSubsequences(self, s: str) -> List[str]:
        result = []

        def backtrack(start, path):
            if path:
                result.append(''.join(path))
            for i in range(start, len(s)):
                path.append(s[i])
                backtrack(i + 1, path)
                path.pop()

        backtrack(0, [])
        return sorted(result)
