print("importing...")
from deepface import DeepFace
print("imported!")
embedding_objs = DeepFace.represent(img_path = "/home/dietpi/pffrocd/lfw/Jose_Santos/Jose_Santos_0001.jpg", model_name="SFace")
embedding = embedding_objs[0]["embedding"]
print("f{embedding=}")
