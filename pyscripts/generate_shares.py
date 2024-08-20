#!/usr/bin/env python3
""" Simple script that extracts the shares from the given image embedding"""
import time
# Start the timer
start_time = time.time()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow warnings https://stackoverflow.com/a/40871012
import argparse
from deepface import DeepFace
import pffrocd
import numpy as np
import quantization as qt

# Create the argument parser
parser = argparse.ArgumentParser(description='Extract face embedding from an image')
parser.add_argument('-i', '--input', type=str, help='Input shares file path', required=True)
parser.add_argument('-b', '--byte', type=int, help='The big length', required=True)
parser.add_argument('-q', '--quantize', type=bool, help='Quantize the embeddings', default=False, required=False)
parser.add_argument('-o', '--output', type=str, help='Output file path', required=True)
parser.add_argument('-s', '--scalar', type=str, help='Output for scalar', required=False)
args = parser.parse_args()

bit_length = args.byte
quantize = args.quantize

if bit_length == 64:
    NUMPY_DTYPE = np.float64
elif bit_length == 32:
    NUMPY_DTYPE = np.float32
elif bit_length == 16:
    NUMPY_DTYPE = np.float16
else:
    raise Exception("Invalid bit length")

# read embeddings from input file
input_file = args.input
with open(input_file, 'r') as f:
    ref_img_embedding = [float(line.strip()) for line in f]

if quantize:
    ref_img_embedding = qt.scalar_quantisation_percentile(ref_img_embedding)

    norm_ref_img_embedding = [np.linalg.norm(ref_img_embedding)]
    share0scalar_y, share1scalar_y = pffrocd.create_shares(np.array(norm_ref_img_embedding, dtype=NUMPY_DTYPE), NUMPY_DTYPE, False)
    # prepare the scalar shares
    share0scalar_y = np.array([share0scalar_y[0]], dtype=NUMPY_DTYPE)
    share1scalar_y = np.array([share1scalar_y[0]], dtype=NUMPY_DTYPE)

share0, share1 = pffrocd.create_shares(np.array(ref_img_embedding, dtype=NUMPY_DTYPE), dtype=NUMPY_DTYPE, quantized=quantize)
if quantize:
    array = []
    for i in range(len(share0)):
        val = np.int32(int(share0[i]))
        array.append(val)
    share0 = array
    array = []
    for i in range(len(share1)):
        val = np.int32(int(share1[i]))
        array.append(val)
    share1 = array

    share0 = np.array(share0, dtype=np.int32)
    share1 = np.array(share1, dtype=np.int32)    
# Write the share to the output file
output_file = args.output
with open(output_file, 'w') as f:
    for i in share1:
        f.write(f"{i}\n")

if quantize:
    output_file = args.scalar
    with open(output_file, 'w') as f:
        for i in share1scalar_y:
            f.write(f"{i}\n")

if quantize:
    # Print share1 so it is brought back to master where it can be send to the client. 
    print("%s | %s" % (share0scalar_y, share0))
else:
    print(share0)
