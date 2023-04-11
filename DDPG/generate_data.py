import numpy as np

list = np.append(1,2)
list = np.append(list,3)
list = np.append(list,4)

copy = list.copy()
print(copy)

copy[3] = 8
print(copy)

another = list[2:]
print(list)
print(another)