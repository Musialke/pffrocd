#!/usr/bin/env python3
import pffrocd # helper functions
import numpy as np

rng = np.random.default_rng()

pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_randint_nscen_nsimd'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"

# get two random int vectors
x = rng.integers(0, 30, 128)
y = rng.integers(0, 30, 128)


output = pffrocd.run_sfe(x, y)

print("X:")
print(x)

print("Y:")
print(y)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))

print(output.stdout)

