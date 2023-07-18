#!/usr/bin/env python3
import pffrocd # helper functions
import numpy as np
import random
import time
import sys


pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_float_scen_simd'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"

fxor = lambda x,y:(x.view("int64")^y.view("int64")).view("float64")

def generate_nonce(a):
    n = np.zeros(128)
    for i in range(len(a)):
        x = np.double(np.random.uniform(-3,3))
        n_i = fxor(a[i], x)
        while np.isnan(n_i):
            print(f"{i=}, {n_i=}")
            x = np.double(np.random.uniform(-3,3))
            n_i = fxor(a[i], x)
        n[i] = n_i
    return n


same_person = [True] * 10 + [False] * 10


for sp in same_person:
    # get two embeddings of different people
    x, y = pffrocd.get_two_random_embeddings(sp)

    r = generate_nonce(y)

    print(f"size of embedding:{sys.getsizeof(x[0])}")

    print(f"size of nonce:{sys.getsizeof(r[0])}")

    y1 = r

    y0 = fxor(y, y1)


    output = pffrocd.run_sfe(x, y, y_0=y0, y_1=y1)

    print(output.stdout)
    with open("sample_aby_output.txt", "w") as f:
        f.write(output.stdout)
    

    print("NUMPY COS_DIST:")
    print(pffrocd.get_cos_dist_numpy(x,y))

    time.sleep(10)

# print("EMBEDDING X:")
# print(x)

# print("EMBEDDING Y:")
# print(y)

# print("Y1:")
# print(y1)

# print("Y0:")
# print(y0)

# print("Y0 XOR Y1:")
# print(fxor(y0, y1))

# print("Y0 XOR Y1 == Y:")
# print(fxor(y0, y1) == y)

# print("NUMPY COS_DIST:")
# print(pffrocd.get_cos_dist_numpy(x,y))


