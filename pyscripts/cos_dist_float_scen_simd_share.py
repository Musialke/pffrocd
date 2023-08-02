#!/usr/bin/env python3
import pffrocd # helper functions
import numpy as np
import random
import time
import sys


pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_float_scen_simd_share'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"
    


# get two embeddings of different people
x, y = pffrocd.get_two_random_embeddings(False)

r = pffrocd.generate_nonce(y)

y1 = r

y0 = pffrocd.fxor(y, y1)


output = pffrocd.run_sfe(x, y, y_0=y0, y_1=y1)

print(output.stdout)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))

print("EMBEDDING X:")
print(x)

print("EMBEDDING Y:")
print(y)

# print("Y1:")
# print(y1)

# print("Y0:")
# print(y0)

# print("Y0 XOR Y1:")
# print(fxor(y0, y1))

# print("Y0 XOR Y1 == Y:")
# print(fxor(y0, y1) == y)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))


