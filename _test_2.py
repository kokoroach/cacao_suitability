# from itertools import groupby

# def len_iter(items):
#     return sum(1 for _ in items)

# def consecutive_one(data):
#     if all(v == 0 for v in data):
#         return 0
#     return max(len_iter(run) for val, run in groupby(data) if val)

# data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# r = consecutive_one(data)
# # print(groupby(data))
# print(r)
_lists = [ [1,2,3], [4,5,6], [7,8,9], [1,2,3], [4,5,6], [7,8,9]]
import numpy
import csv
# a = numpy.asarray()
# numpy.savetxt("foo.csv", a, fmt="%d", delimiter=",")

file_out = 'test.csv'

add = []
with open(file_out, 'w', newline='') as csvfile_out:
    writer = csv.writer(csvfile_out, delimiter=',')

    for i, _list in enumerate(_lists):
        add.append(_lists[i])
        if i%3 == 0:
            print('here')
            writer.writerows(add)
            add = []
    writer.writerows(add)
    


