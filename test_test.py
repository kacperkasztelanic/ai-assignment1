from data_loader import load
import numpy as np
import random
from main import Assignment

TEST_DATA_1 = load('had20.dat')

# n, matrix_flow, matrix_distance = TEST_DATA_1
# print(n, matrix_flow.shape)
# print(matrix_flow.shape[1])
# print(sum(np.array([2, 5, 3]) == np.array([2, 5, 3])))

# random.seed(1)
# print(random.randrange(3), random.randrange(3))
# random.seed(1)
# print(random.randrange(3), random.randrange(3))

# random.seed(1)
# # print(random.sample(range(4), 1))
# list_not_contains = [3, 4, 7]
# random.shuffle(list_not_contains)
# print(list_not_contains)

# x = [2,1,4, 0]
# x = np.repeat(x,4).reshape((4,4))
# print(x)
#
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[[0,0], [1,1], [[1,1] [0,1]]]])
# print(b(a))

#                 a  b  c  d  e  f
A = np.array([[0, 1, 2, 3, 4, 5],
              [1, 0, 3, 4, 5, 6],
              [2, 3, 0, 5, 6, 7],
              [3, 4, 5, 0, 7, 8],
              [4, 5, 6, 7, 0, 9],
              [5, 6, 7, 8, 9, 0]])

#            a  d  b  e  c  f
new_order = [0, 3, 1, 4, 2, 5]
A1 = A[new_order][:, new_order]
print(A1)

print(np.arange(5))
