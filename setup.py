from setuptools import setup
import os
from io import open # Py2.7 compatibility

def readme():
    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'README.md'
            ), encoding='utf8') as fp:
        return fp.read()

setup(
    name = 'aaa-approx',
    version = '1.0.1',
    description = 'A Python implementation of the AAA algorithm for rational approximation',
    long_description = readme(),
    long_description_content_type = 'text/markdown',
    author = 'Clemens Hofreither',
    author_email = 'chofreither@numa.uni-linz.ac.at',
    url = 'https://github.com/c-f-h/aaa',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: BSD License',
    ],
    py_modules = ['aaa'],
    install_requires = [
        'numpy>=1.11',
        'scipy',
    ],
    tests_require = 'nose',
    test_suite = 'nose.collector'
)
