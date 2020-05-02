from mpi4py import MPI
import numpy as np
from orphics import stats
# example: https://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

x = np.arange(7)
list = np.array_split(x, 4)
print(list)
print("my rank is %d:" % rank)
data = np.zeros(2)
for i in list[rank]:
    print(i)
    data[i] = i
print(data)
# np.savetxt(f'rank{rank}.txt',data)
