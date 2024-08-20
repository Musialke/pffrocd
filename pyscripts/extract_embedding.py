#!/usr/bin/env python3
""" Simple script that extracts the embedding from the given image"""
import time
# Start the timer
start_time = time.time()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow warnings https://stackoverflow.com/a/40871012
import argparse
from deepface import DeepFace
import numpy as np

# Create the argument parser
parser = argparse.ArgumentParser(description='Extract face embedding from an image')
parser.add_argument('-i', '--input', type=str, help='Input image file path', required=True)
parser.add_argument('-o', '--output', type=str, help='Output file path', required=True)
parser.add_argument('-n', '--normalize', type=bool, help='Normalize the embedding', required=False, default=False)
args = parser.parse_args()

# Load the image
input_image = args.input

# Extract the face embedding
embedding = DeepFace.represent(input_image, model_name='SFace', enforce_detection=False)[0]["embedding"]

should_normalize = args.normalize
if should_normalize:
    embedding = embedding / np.linalg.norm(embedding)

# Write the embedding to the output file
output_file = args.output
with open(output_file, 'w') as f:
    for i in embedding:
        f.write(f"{i}\n")

total_time = time.time() - start_time
print(total_time)