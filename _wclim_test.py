import numpy as np
from math import pow

# data = [90, 65, 75, 40, 55]
data = [100, 100, 100, 100, 0]
# data = [1, 229, 23.083, 25.858, 28.658]

result = 0

def get_land_index(iterable):
    n = len(iterable)
    RS = get_RS(data)
    LS = get_LS(iterable)
    return LS * RS

def get_RS(iterable):
    a = np.array(iterable)
    r = a.prod(dtype=np.int64)/pow(100, 5)
    return pow(r, 1/len(iterable))

def get_LS(iterable):
    n = len(iterable)

    result = 0
    for i, j in enumerate(iterable):
        if i == 0:
            result = pow(j, 1/n)
        else:
            result *= pow(j, 1/n)
    return result


if __name__ == '__main__':
    r = get_land_index(data)
    print(r)