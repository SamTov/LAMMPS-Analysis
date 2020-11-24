import setuptools
from distutils.core import setup, Extension
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MDSuite",
    version="0.0.2",
    author="Samuel Tovey",
    author_email="tovey.samuel@gmail.com",
    description="A postprocessing tool for molecular dynamics simulations targeting the machine learning community.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamTov/MDSuite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['seaborn',
                      'h5py',
                      'numpy',
                      'matplotlib',
                      'scipy',
                      'alive_progress',
                      'psutil',
                      'mendeleev'],
    ext_modules=cythonize("mdsuite/cython_extensions/convolution.pyx")
)
