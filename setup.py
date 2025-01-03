from setuptools import setup, find_packages

setup(
    name='pyfel1d',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'numexpr',
        'joblib',
        'mpmath',
        'numba',
    ],
)