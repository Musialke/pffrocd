import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # https://stackoverflow.com/a/40871012
from deepface import DeepFace
import argparse


parser = argparse.ArgumentParser(
                    prog='Extract Embedding',
                    description='Extracts the face features vector from a given image')

parser.add_argument('imagepath')
parser.add_argument('--output', help="output file")
args = parser.parse_args()
embedding_objs = DeepFace.represent(img_path = args.imagepath, model_name="SFace", enforce_detection=True)
embedding = embedding_objs[0]["embedding"]
assert isinstance(embedding, list)
if (args.output):
    with open(args.output, 'w') as f:
        for v in embedding:
            f.write(f"{v}, ")
else: 
    print(embedding)

