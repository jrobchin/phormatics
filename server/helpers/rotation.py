
import numpy

def rotation(a):
    b = [[0,-1],[1,0]]
    c = (1,0) - numpy.matmul(a,b)
    return(c)