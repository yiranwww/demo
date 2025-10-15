class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        hours = [8, 4, 2, 1]
        minutes = [32, 16, 8, 4, 2, 1]
        res = []

        def backtrack(num_left, start, hour, minute):
            if num_left == 0:
                if hour < 12 and minute < 60:
                    res.append(f"{hour}:{minute:02d}")
                return 
            for i in range(start, 10):
                if i < 4:
                    backtrack(num_left -1, i+1, hour + hours[i], minute)
                else:
                    backtrack(num_left -1, i +1, hour, minute + minutes[i-4])
                
        backtrack(turnedOn, 0, 0, 0)
        return res