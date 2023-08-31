#!/usr/bin/env python3

"""Test the cos_dist function locally with multiple runs, resetting the ABY party after each run"""

NR_OF_RUNS = 20
ABS_PATH = "/home/kamil/Documents/uni/thesis/pffrocd"
import json
import subprocess
import pffrocd
import pandas as pd

# get some random images

imgs = pffrocd.get_random_images_except_person(root_dir='lfw', excluded_person=None, num_images=NR_OF_RUNS+1)

# set the first image as the 'reference' image (registered at the service provider) and remove it from the list of images
ref_img = imgs[0]
imgs = imgs[1:]

# create shares of the reference image
ref_img_embedding = pffrocd.get_embedding(ref_img)
share0, share1 = pffrocd.create_shares(ref_img_embedding)

#write shares to file
with open(f"{ABS_PATH}/ABY/build/bin/share0.txt", 'w') as file:
    for i in share0:
        file.write(f"{i}\n")

with open(f"{ABS_PATH}/ABY/build/bin/share1.txt", 'w') as file:
    for i in share1:
        file.write(f"{i}\n")

# write all embeddings of the images to files
for i, img in enumerate(imgs):
    print(f"({i+1}/{NR_OF_RUNS}) writing embeddings for {img}")
    img_embedding = pffrocd.get_embedding(img)
    with open(f"{ABS_PATH}/ABY/build/bin/embeddings{i}.txt", 'w') as file:
        for x_i, y_i in zip(img_embedding, ref_img_embedding):
            file.write(f"{x_i} {y_i}\n")

# execute the ABY cos dist computation
CMD = f"cd ABY/build/bin ; ./cos_dist_float_scen_simd -r 1 -f embeddings -o {ABS_PATH} -d {NR_OF_RUNS} & (./cos_dist_float_scen_simd -r 0 -f embeddings -o {ABS_PATH} -d {NR_OF_RUNS} 2>&1 > /dev/null)"
output = subprocess.run(CMD, shell=True, capture_output=True, text=True)
assert (output.returncode == 0), f"{output.stdout=}, {output.stderr=}" # make sure the process executed successfully
# print(output.stdout)
# parse the outputs
l = []
benchmarks = output.stdout.split("split here")
for b in benchmarks:
    parsed = pffrocd.parse_aby_output(b)
    if parsed:
        l.append(parsed)

# for x in l:
#     print(json.dumps(x, sort_keys=False, indent=4))
df = pd.DataFrame(l)
df.to_csv("local_reset.csv", index=False)