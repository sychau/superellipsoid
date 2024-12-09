from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension module for Cython
ext_modules = [
    Extension(
        "fast_sampler._sampler",
        [
            "fast_sampler/_sampler.pyx",
            "fast_sampler/sampling.cpp"
        ],
        language="c++11",
        
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11", "-O3"]
    )
]

# Run the setup to build the extension
setup(
    name="fast_sampler",
    ext_modules=cythonize(ext_modules),  # Compile all Cython extensions
)
