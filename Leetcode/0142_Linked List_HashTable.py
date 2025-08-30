# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head):
        dummy = head
        seen = set()

        while dummy:
            if dummy in seen:
                return dummy
            seen.add(dummy)
            dummy = dummy.next
        return None