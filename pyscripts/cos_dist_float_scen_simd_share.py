#!/usr/bin/env python3
print("importing pffrocd...")
import pffrocd # helper functions
print("pffrocd imported!")
import numpy as np
import random
import time
import quantization as qt
import sys
import os

pffrocd.EXECUTABLE_PATH = "ABY/build/bin"
pffrocd.EXECUTABLE_NAME = 'cos_dist_float_scen_simd_32'
pffrocd.INPUT_FILE_NAME = f"input_{pffrocd.EXECUTABLE_NAME}.txt"
pffrocd.OUTPUT_FILE_NAME = f"/home/chiem/pffrocd"
NUMPY_DTYPE = np.float32

def cosine_similarity(v1, v2):
    # Compute the cosine similarity
    return 1- np.dot(v1, v2)

def get_two_random_images(same_person):
    """Get two random embeddings of either the same person or two different people out of all the images available"""
    people = [p for p in os.listdir('lfw') if os.path.isdir(os.path.join('lfw', p))] # list of all people that have images
    people_with_multiple_images = [p for p in people if len([img for img in os.listdir(os.path.join("lfw", p)) if img != '.DS_Store']) > 1]  # list of people with more than one image in folder
    img1, img2 = None, None # face embeddings
    while img1 is None or img2 is None: # try until the chosen images have detectable faces
        try:
            if same_person:
                # same person should have more than one image (we might still end up choosing the same image of that person with prob 1/n, but that's ok)
                person1 = random.choice(people_with_multiple_images)
                person2 = person1
            else:
                # two persons chosen should be different
                person1 = random.choice(people)
                person2 = random.choice([p for p in people if p != person1])
            # get two random images
            img1 = f"lfw/{person1}/{random.choice(os.listdir(f'lfw/{person1}'))}"
            img2 = f"lfw/{person2}/{random.choice(os.listdir(f'lfw/{person2}'))}"
        except Exception as e:
            # failed to detect faces in images, try again
            # print(e)
            pass

    return img1,img2

img1, img2 = get_two_random_images(False)
x = pffrocd.get_embedding(img1, dtype=np.float32)
y = pffrocd.get_embedding(img2, dtype=np.float32)
x = x / np.linalg.norm(x)
y = y / np.linalg.norm(y)

share0, share1 = pffrocd.create_shares(np.array(x, dtype=NUMPY_DTYPE), NUMPY_DTYPE, False)
share0prime, share1prime = pffrocd.create_shares(np.array(y, dtype=NUMPY_DTYPE), NUMPY_DTYPE, False)

# share0 = np.array(share0, dtype=np.int32)
# share1 = np.array(share1, dtype=np.int32)
# share0prime = np.array(share0prime, dtype=np.int32)
# share1prime = np.array(share1prime, dtype=np.int32)
# x = np.array(x, dtype=np.int32)
# y = np.array(y, dtype=np.int32)
output = pffrocd.run_sfe_improved(x, y, y_0=share0, y_1=share1, x_0=share0prime, x_1=share1prime)
print(output.stdout)

print("NUMPY COS_DIST:")
print(pffrocd.get_cos_dist_numpy(x,y))
# # the dot product written out
# sum = 0
# for i in range(0, len(x)):
#     #print(f"x[{i}]: {x[i]} * y[{i}]: {y[i]} = {x[i]*y[i]}")
#     sum+=x[i]*y[i]
# print(sum)
