class MyQueue:

    def __init__(self):
        self.stack = []
        self.aux = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        

    def pop(self) -> int:
        while self.stack:
            self.aux.append(self.stack.pop())
        res = self.aux.pop()
        while self.aux:
            self.stack.append(self.aux.pop())
        return res

        

    def peek(self) -> int:
        return self.stack[0]
        

    def empty(self) -> bool:
        return len(self.stack) == 0
        


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()