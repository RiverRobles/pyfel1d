# pyfel1d
A package for simulating 1D free-electron laser physics in Python. 

This package is similar to and definitely inspired by zfel (https://github.com/slaclab/zfel), with slight algorithmic differences. I wrote this because I wanted a more general 1D FEL simulation code capable of dealing with complicated effects like arbitrary bunch profiles, seed fields, tracking of harmonic radiation fields, and more. 

The basic FEL simulation code is contained within fel.py. The code uses scaled units described in the documentation. Conversion between scaled and unscaled units is handled by the class defined in converter.py. Particle loading with proper shot noise is handled by the functions within particles.py.

I have included many example Jupyter notebooks tailored to different use cases. 