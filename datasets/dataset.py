
class MyData():
    def __init__(self):
        self.nums = [1,2,3,4,5,6,7,8]

    def __getitem__(self, item):
        return self.nums[item]

    def __len__(self):
        return len(self.nums)


mydata = MyData()
print(mydata[1])
print(len(mydata))