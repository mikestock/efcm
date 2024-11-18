cimport numpy as cnp
import numpy as np
from libc.math cimport sin, cos, M_PI, sqrt
from libc.stdlib cimport malloc, free
import struct

def droop( cnp.ndarray[float, ndim=1] arr, 
			float tau,
			float z=0, 
			float w=0):
	"""droop applies an exponential decay to the data
	
	for simplicity, tau here should be in samples
	
	to droop the data, we apply a first order high pass filter.  This 
	is done using an iterative algorithm.  You should look up how this 
	works, it's cool, and super efficient.
	"""
	
	###
	# initial definitions
	cdef float alpha = tau/(1.+tau)	#this is a parameter for doing the lpf
	cdef int N = arr.shape[0]	#the length of the input array
	cdef int i
	
	###
	# create output
	cdef cnp.ndarray[float, ndim=1] output = np.empty( N, dtype='f' )
	
	for i in range( N ):
		output[i] = alpha*w + alpha*(arr[i]-z)
		#we do it this way so that we don't ever index negative numbers
		z = arr[i]
		w = output[i]
	
	return output

def dedroop(cnp.ndarray[float, ndim=1] arr, 
			float tau,
			float z=0, 
			float w=0):

	"""dedroop undoes the exponential decay that is applied to our 
	signal by the fast antenna.  
	
	for simplicity, tau here should be in samples
	
	to dedroop, we invert the recurrence relation for the hpf
	"""

	###
	# initial definitions
	cdef float alpha = tau/(1.+tau)	#this is a parameter for doing the lpf
	cdef int N = arr.shape[0]	#the length of the input array
	cdef int i

	

	###
	# create output
	cdef cnp.ndarray[float, ndim=1] output = np.empty( N, dtype='f' )
	
	for i in range(N):
		output[i] = (arr[i]-alpha*z)/alpha + w
		#we do it this way so that we don't even index negative numbers
		z = arr[i]
		w = output[i]
	
	return output

def lpf( cnp.ndarray[float, ndim=1] arr, 
				float tau, 
				float z=0 ):
		"""firstorder( arr, tau, z=0 )
		First order RC filter
		
		tau 	in samples
		z 		a memory parameter so that the filter can be applied 
				to arrays longer than memory"""
		
		###
		# initial definitions
		cdef float alpha = 1./(1.+tau) 
		cdef int N = arr.shape[0]
		cdef int i

		###
		# create output
		cdef cnp.ndarray[float, ndim=1] output = np.empty( N, dtype='f' )
		
		for i in range(N):
			arr[i] = alpha*arr[i]+(1-alpha)*z 
			z=arr[i]
		return arr

def hpf( cnp.ndarray[float, ndim=1] arr, 
			float tau,
			float z=0, 
			float w=0):
	"""droop applies an exponential decay to the data
	
	for simplicity, tau here should be in samples
	
	to droop the data, we apply a first order high pass filter.  This 
	is done using an iterative algorithm.  You should look up how this 
	works, it's cool, and super efficient.
	"""
	
	###
	# initial definitions
	cdef float alpha = tau/(1.+tau)	#this is a parameter for doing the lpf
	cdef int N = arr.shape[0]	#the length of the input array
	cdef int i
	
	###
	# create output
	cdef cnp.ndarray[float, ndim=1] output = np.empty( N, dtype='f' )
	
	for i in range( N ):
		output[i] = alpha*w + alpha*(arr[i]-z)
		#we do it this way so that we don't ever index negative numbers
		z = arr[i]
		w = output[i]
	
	return output
