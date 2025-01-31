# Before running, apt-get install cython
# to compile the cintf_tools.so module:
# python **** build_ext --inplace
# if you get the error:
# "ValueError: numpy.ufunc has the wrong size, try recompiling"
# this is an indication that cintf_tools was not properly compiled for 
# your architecture
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
 
import numpy
 
ext = Extension("filters", ["filters.pyx"], 
	include_dirs = [numpy.get_include()])
 
setup(ext_modules=[ext], cmdclass = {'build_ext': build_ext})
