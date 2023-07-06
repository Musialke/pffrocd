import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # https://stackoverflow.com/a/40871012
from deepface import DeepFace
import argparse


parser = argparse.ArgumentParser(
                    prog='Extract Embedding',
                    description='Extracts the face features vectors from two images to be compared')

parser.add_argument('imagepath1')
parser.add_argument('imagepath2')
parser.add_argument('--output', help="output file")
args = parser.parse_args()
embedding1 = DeepFace.represent(img_path = args.imagepath1, model_name="SFace", enforce_detection=True)[0]["embedding"]
embedding2 = DeepFace.represent(img_path = args.imagepath2, model_name="SFace", enforce_detection=True)[0]["embedding"]
assert isinstance(embedding1, list)
assert isinstance(embedding2, list)
if (args.output):
    with open(args.output, 'w') as f:
        for x, y in zip(embedding1, embedding2):
            f.write(f"{x} {y}\n")
else: 
    print(embedding1, embedding2)

