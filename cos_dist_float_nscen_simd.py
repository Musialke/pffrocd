#!/usr/bin/env python3
import pffrocd # helper functions
import numpy as np

pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_float_scen_simd_32'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"
pffrocd.OUTPUT_FILE_NAME = "/home/chiem/pffrocd"

# get two embeddings of different people
x,y=pffrocd.get_two_random_embeddings(False)
#convert x to np.ndarray
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)
share0, share1 = pffrocd.create_shares(y, dtype=np.float32)
# share0prime, share1prime = pffrocd.create_shares(share0, dtype=np.float32)

print("share0:", share0)

# Print out the cosine distance for verificaiton before the normalization happens
print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))

# # Normalize the embeddings
# x = x / np.linalg.norm(x)
# y = y / np.linalg.norm(y)

# # Print out the cosine distance for verificaiton after the normalization happens
# print("Normalize COS_DIST:")
# print(1 - np.dot(x, y))

# Run the circuit
output = pffrocd.run_sfe(x, y, y_0=share0, y_1=share1)

# Print the output
print(output.stdout)