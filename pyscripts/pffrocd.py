"""
Helper function for scripts running the face verification with sfe
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow warnings https://stackoverflow.com/a/40871012
import numpy as np
import subprocess
import random
import re
from deepface import DeepFace
from fabric import Connection


EXECUTABLE_PATH = None
INPUT_FILE_NAME = None
EXECUTABLE_NAME = None

def get_cos_dist_numpy(x, y):
    """
    Compute the cosine distance between two vectors using numpy
    """
    return 1 - np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

def run_sfe(x, y, y_0=None, y_1=None):
    """
    Write the vectors to files used by ABY executable
    If y_0 and y_1 are provided run it as actual scenario (shared IN gates)
    Otherwise run as test providing two vectors to be compared
    """
    with open(f"{EXECUTABLE_PATH}/{INPUT_FILE_NAME}", 'w') as f:
        for x_i, y_i in zip(x, y):
            f.write(f"{x_i} {y_i}\n")
            
    if y_0 is not None and y_1 is not None:
        # write the shares into separate files
        with open(f"{EXECUTABLE_PATH}/share0.txt", 'w') as f:
            for i in y_0:
                f.write(f"{i}\n")
        with open(f"{EXECUTABLE_PATH}/share1.txt", 'w') as f:
            for i in y_1:
                f.write(f"{i}\n")
            
    # execute the ABY cos sim computation
    CMD = f"./{EXECUTABLE_NAME} -r 0 -f {INPUT_FILE_NAME} & (./{EXECUTABLE_NAME} -r 1 -f {INPUT_FILE_NAME} 2>&1 > /dev/null)"
    output = subprocess.run(CMD, shell=True, capture_output=True, text=True, cwd=EXECUTABLE_PATH)
    assert (output.returncode == 0), f"{output.stdout=}, {output.stderr=}" # make sure the process executed successfully
    return output

def get_embedding(imagepath):
    return DeepFace.represent(img_path = imagepath, model_name="SFace", enforce_detection=True)[0]["embedding"]

def get_two_random_embeddings(same_person):
    print(os.getcwd())
    people = os.listdir('lfw') # list of all people that have images
    people_with_multiple_images = [p for p in people if len(os.listdir(f"lfw/{p}")) > 1] # list of people with more than one image in folder
    embedding1, embedding2 = None, None # face embeddings
    while embedding1 is None or embedding2 is None: # try until the chosen images have detectable faces
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
            # try to extract embeddings from both images
            embedding1 = get_embedding(img1)
            embedding2 = get_embedding(img2)
        except Exception as e:
            # failed to detect faces in images, try again
            # print(e)
            pass
    return np.array(embedding1), np.array(embedding2)

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


def parse_aby_output(s):
    """Parses the benchmark output of ABY and returns stats of interest in a dictionary"""

    # get all numbers from the output string
    numbers = re.findall(r"[-+]?(?:\d*\.*\d+)", s) 

    # prepare dictionary
    d = {'online_time': {}, 'complexities': {}, 'communication': {}}

    # online_time
    some_keys = ['bool', 'yao', 'yao_rev', 'arith', 'splut']
    a_dict = {'local_gates': '', 'interactive_gates': '', 'layer_finish': ''}
    d['online_time'] = {key : a_dict.copy() for key in some_keys}

    d['online_time'] |= {'communication': ''}

    d['online_time']['bool']['local_gates'] = numbers[0]
    d['online_time']['bool']['interactive_gates'] = numbers[1]
    d['online_time']['bool']['layer_finish'] = numbers[2]

    d['online_time']['yao']['local_gates'] = numbers[3]
    d['online_time']['yao']['interactive_gates'] = numbers[4]
    d['online_time']['yao']['layer_finish'] = numbers[5]

    d['online_time']['yao_rev']['local_gates'] = numbers[6]
    d['online_time']['yao_rev']['interactive_gates'] = numbers[7]
    d['online_time']['yao_rev']['layer_finish'] = numbers[8]

    d['online_time']['arith']['local_gates'] = numbers[9]
    d['online_time']['arith']['interactive_gates'] = numbers[10]
    d['online_time']['arith']['layer_finish'] = numbers[11]

    d['online_time']['splut']['local_gates'] = numbers[12]
    d['online_time']['splut']['interactive_gates'] = numbers[13]
    d['online_time']['splut']['layer_finish'] = numbers[14]

    d['online_time']['communication'] = numbers[15]


    # complexities
    d['complexities'] = {'boolean_sharing': {'ands': numbers[16], 'depth': numbers[18]}}
    d['complexities'] |= {'total_vec_and': numbers[19], 'total_non_vec_and': numbers[20], 'xor_vals': numbers[21], 'gates': numbers[22],'comb_gates': numbers[23],'combstruct_gates': numbers[24], 'perm_gates': numbers[25], 'subset_gates': numbers[26], 'split_gates': numbers[27]}
    d['complexities'] |= {'yao':{'ands':numbers[28], 'depth':numbers[29]}}
    d['complexities'] |= {'reverse_yao':{'ands':numbers[30], 'depth':numbers[31]}}
    d['complexities'] |= {'arithmetic_sharing':{'muls':numbers[32], 'depth':numbers[33]}}
    d['complexities'] |= {'sp_lut_sharing':{'ot_gates_total':numbers[34], 'depth':numbers[35]}}
    d['complexities'] |= {'total_nr_of_gates':numbers[36],'total_depth':numbers[37]}

    # timings
    d['timings'] = {'total': numbers[38], 'init': numbers[39], 'circuitgen': numbers[40], 'network': numbers[41], 'baseots': numbers[42], 'setup': numbers[43], 'otextension':numbers[44], 'garbling':numbers[45], 'online': numbers[46]}

    # communication
    some_keys = ['total', 'base_ots', 'setup', 'otextension', 'garbling', 'online']
    a_dict = {'sent':'', 'received':''}
    d['communication'] = {key: a_dict.copy() for key in some_keys}

    d['communication']['total']['sent'] = numbers[47]
    d['communication']['total']['received'] = numbers[48]

    d['communication']['base_ots']['sent'] = numbers[49]
    d['communication']['base_ots']['received'] = numbers[50]

    d['communication']['setup']['sent'] = numbers[51]
    d['communication']['setup']['received'] = numbers[52]

    d['communication']['otextension']['sent'] = numbers[53]
    d['communication']['otextension']['received'] = numbers[54]

    d['communication']['garbling']['sent'] = numbers[55]
    d['communication']['garbling']['received'] = numbers[56]

    d['communication']['online']['sent'] = numbers[57]
    d['communication']['online']['received'] = numbers[58]

    d['cos_dist_ver'] = numbers[59]
    d['cos_dist_sfe'] = numbers[60]

    return d



def ssh_to_client(host, user, password, executable):
    """
    Connects to the client via SSH and runs the specified executable.

    Args:
        host (str): The hostname of the client.
        user (str): The username on the client.
        password (str): The password for the user on the client.
        executable (str): The path to the executable to run on the client.

    Returns:
        str: The output of the executable.
    """

    with Connection(host, user=user, password=password) as conn:
        output = conn.run(executable)
        return output

if __name__ == "__main__":
    host = "client.example.com"
    user = "pi"
    password = "raspberry"
    executable = "/home/pi/secure_function_evaluation"

    output = ssh_to_client(host, user, password, executable)
    print(output)