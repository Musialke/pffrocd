#!/usr/bin/env python3
import pffrocd # helper functions
import numpy as np

pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_int_nscen_simd'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"
FACTOR = 10 # factor 100 causes the results to be wrong! (overflow?)

# get two embeddings of different people
x, y = pffrocd.get_two_random_embeddings(False)

print(f"COS_DIST OF ORIGINAL VECTORS = {pffrocd.get_cos_dist_numpy(x,y)}")

# multiply the floats by some factor and then get rid of the remainder after the decimal value

x = (x * FACTOR).round().astype(int)
y = (y * FACTOR).round().astype(int)

# upscale elements to non-negative
add_min = min(x.min(), y.min())

x = x + abs(add_min)
y = y + abs(add_min)

output = pffrocd.run_sfe(x, y)


print("EMBEDDING X:")
print(x)

print("EMBEDDING Y:")
print(y)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))

print(output.stdout)

