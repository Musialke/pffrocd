#!/usr/bin/env python3
import pffrocd # helper functions
import numpy as np

rng = np.random.default_rng()

pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_randint_scen_nsimd'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"

# get two random int vectors
x, y = pffrocd.get_two_random_embeddings(False)

# multiply the floats by some factor and then get rid of the remainder after the decimal value

x = rng.integers(-30, 30, 128)
y = rng.integers(-30, 30, 128)

# create the shares

r = rng.integers(-30, 30, 128) # nonces

y_1 = r # drone's share, nonces

y_0 = np.bitwise_xor(y, y_1) # md's share

output = pffrocd.run_sfe(x, y, y_0=y_0, y_1=y_1)

print("X:")
print(x)

print("Y:")
print(y)

print("Y_0:")
print(y_0)

print("Y_1:")
print(y_1)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))

print(output.stdout)

