#!/usr/bin/env python3
import pffrocd # helper functions
import numpy as np

pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_float_nscen_nsimd'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"

# get two embeddings of different people
x, y = pffrocd.get_two_random_embeddings(False)

output = pffrocd.run_sfe(x, y)


print("EMBEDDING X:")
print(x)

print("EMBEDDING Y:")
print(y)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))

print(output.stdout)

