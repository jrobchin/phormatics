import math 
import numpy

def is_hip_higher(a, h, s):
	v = ((s[0]-a[0]),(s[1]-a[1]))
	return (h[1] > a[1] + ((v[1] * h[0]) / v[0]) )  
