print("importing...")
from deepface import DeepFace
print("imported!")
embedding_objs = DeepFace.represent(img_path = "lfw/George_W_Bush/George_W_Bush_0001.jpg", model_name="SFace")
embedding = embedding_objs[0]["embedding"]
print(f"{embedding=}")
