#!/usr/bin/env python3
print("importing pffrocd...")
import pffrocd # helper functions
print("pffrocd imported!")
import numpy as np
import random
import time
import sys


pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_float_scen_simd'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"



# get two embeddings of different people
print("Getting two random embeddings...")
x, y = pffrocd.get_two_random_embeddings(False)
print("got the embeddings!")

r = pffrocd.generate_nonce(y)

y1 = r

y0 = pffrocd.fxor(y, y1)


output = pffrocd.run_sfe(x, y, y_0=y0, y_1=y1)

print(output.stdout)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))
