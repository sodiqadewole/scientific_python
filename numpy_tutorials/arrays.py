import numpy as np

x = np.array([[1,2,3], [4,5,6]], np.int32)
type(x)
x.shape
x.dtype

# indexing
x[1,2]
y = x[:, 1]
y[0] = 9
y
x

np.ones((10,1), order='C').flags.f_contiguous
# Array attributes
x.flags
x.strides # tuple of bytes to step in each dimension when traversing an array
x.ndim # number of dimensions
x.data
x.size
x.itemsize
x.nbytes # total bytes consumed
x.base
x.T
x.real # real part of the array
x.imag # imaginary part of the array
x.flat # A 1-D iterator over the array
x.ctypes

# Array Methods
x.cumsum()
x.cumprod()
x.diagonal()
x.max()
x.min()
x.mean()
x.nonzero()
x.partition()
np.put(x, 0, -1)
x
np.putmask(x, x<0, 0)
x
x.ravel()
np.repeat(5, 3)
np.repeat(x, 3)
np.repeat(x, 3, axis=1)
np.repeat(x, 3, axis=0)
np.swapaxes(x,0,1)
x.T
np.trace(np.eye(3)) # get the sum of the diagonal elements
np.var(x) # compute the variance along an axis/
