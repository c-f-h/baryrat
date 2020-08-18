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
    name = 'baryrat',
    version = '1.2.0',
    description = 'A Python package for barycentric rational approximation',
    long_description = readme(),
    long_description_content_type = 'text/markdown',
    author = 'Clemens Hofreither',
    author_email = 'clemens.hofreither@ricam.oeaw.ac.at',
    url = 'https://github.com/c-f-h/baryrat',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: BSD License',
    ],
    py_modules = ['baryrat'],
    install_requires = [
        'numpy>=1.11',
        'scipy',
    ],
    tests_require = 'nose',
    test_suite = 'nose.collector'
)
