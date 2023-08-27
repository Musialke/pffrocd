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
import paramiko
import logging
import datetime
import json
from pssh.clients import ParallelSSHClient
import pandas as pd
from deepface.commons.distance import findThreshold

logging.getLogger("paramiko").setLevel(logging.WARNING)

threshold = findThreshold("SFace", "cosine")

columns = [
    'ref_img',
    'other_img',
    'result',
    'expected_result',
    'cos_dist_np',
    'cos_dist_sfe',
    'total_time',
    'sfe_time',
    'extraction_time',
    'Command being timed',
    'User time (seconds)',
    'System time (seconds)', 
    'Percent of CPU this job got', 
    'Elapsed (wall clock) time (h:mm:ss or m:ss)', 
    'Average shared text size (kbytes)', 
    'Average unshared data size (kbytes)', 
    'Average stack size (kbytes)', 
    'Average total size (kbytes)', 
    'Maximum resident set size (kbytes)', 
    'Average resident set size (kbytes)', 
    'Major (requiring I/O) page faults', 
    'Minor (reclaiming a frame) page faults', 
    'Voluntary context switches', 
    'Involuntary context switches', 
    'Swaps', 
    'File system inputs', 
    'File system outputs', 
    'Socket messages sent', 
    'Socket messages received', 
    'Signals delivered', 
    'Page size (bytes)', 
    'Exit status',
    'energy_used',
    'online_time.bool.local_gates',
    'online_time.bool.interactive_gates',
    'online_time.bool.layer_finish',
    'online_time.yao.local_gates',
    'online_time.yao.interactive_gates',
    'online_time.yao.layer_finish',
    'online_time.yao_rev.local_gates',
    'online_time.yao_rev.interactive_gates',
    'online_time.yao_rev.layer_finish',
    'online_time.arith.local_gates',
    'online_time.arith.interactive_gates',
    'online_time.arith.layer_finish',
    'online_time.splut.local_gates',
    'online_time.splut.interactive_gates',
    'online_time.splut.layer_finish',
    'online_time.communication',
    'complexities.boolean_sharing.ands',
    'complexities.boolean_sharing.depth',
    'complexities.total_vec_and',
    'complexities.total_non_vec_and',
    'complexities.xor_vals',
    'complexities.gates',
    'complexities.comb_gates',
    'complexities.combstruct_gates',
    'complexities.perm_gates',
    'complexities.subset_gates',
    'complexities.split_gates',
    'complexities.yao.ands',
    'complexities.yao.depth',
    'complexities.reverse_yao.ands',
    'complexities.reverse_yao.depth',
    'complexities.arithmetic_sharing.muls',
    'complexities.arithmetic_sharing.depth',
    'complexities.sp_lut_sharing.ot_gates_total',
    'complexities.sp_lut_sharing.depth',
    'complexities.total_nr_of_gates',
    'complexities.total_depth',
    'timings.total',
    'timings.init',
    'timings.circuitgen',
    'timings.network',
    'timings.baseots',
    'timings.setup',
    'timings.otextension',
    'timings.garbling',
    'timings.online',
    'communication.total.sent',
    'communication.total.received',
    'communication.base_ots.sent',
    'communication.base_ots.received',
    'communication.setup.sent',
    'communication.setup.received',
    'communication.otextension.sent',
    'communication.otextension.received',
    'communication.garbling.sent',
    'communication.garbling.received',
    'communication.online.sent',
    'communication.online.received',
    'cos_dist_ver',
    'cos_dist_sfe'
]

def get_config_in_printing_format(config):
    d = {section: dict(config[section]) for section in config.sections()}
    return json.dumps(
    d,
    sort_keys=False,
    indent=4,
    separators=(',', ': ')
    )


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
    CMD = f"./{EXECUTABLE_NAME} -r 0 -f {INPUT_FILE_NAME} -s 112 -x 0 & (./{EXECUTABLE_NAME} -r 1 -f {INPUT_FILE_NAME} -s 112 -x 0 2>&1 > /dev/null)"
    output = subprocess.run(CMD, shell=True, capture_output=True, text=True, cwd=EXECUTABLE_PATH)
    assert (output.returncode == 0), f"{output.stdout=}, {output.stderr=}" # make sure the process executed successfully
    return output

def get_embedding(imagepath):
    return np.array(DeepFace.represent(img_path = imagepath, model_name="SFace", enforce_detection=False)[0]["embedding"])

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
    """Generates random float nonces given a list of floats of size 128 (the face emedding)
    Checks for nan values after xoring, if that happens then it generates the nonces again
    """
    n = np.zeros(128)
    for i in range(len(a)):
        x = np.double(np.random.uniform(-3,3))
        n_i = fxor(a[i], x)
        while np.isnan(n_i):
            x = np.double(np.random.uniform(-3,3))
            n_i = fxor(a[i], x)
        n[i] = n_i
    return n

def parse_usr_bin_time_output(output):
    """Parses the benchmark output of usr/bin/time and returns stats of interest in a dictionary"""
    metrics = {}
    lines = output.strip().split('\t')

    for line in lines:
        key, value = line.split(': ')
        metrics[key.strip()] = value.strip()
    return metrics


def parse_aby_output(s):
    """Parses the benchmark output of ABY and returns stats of interest in a dictionary"""

    # get all numbers from the output string
    numbers = re.findall(r"[-+]?(?:\d*\.*\d+)", s)

    if not numbers:
        return None

    # prepare dictionary
    d = {}

    try:
    # online_time
        d['online_time.bool.local_gates'] = numbers[0]
        d['online_time.bool.interactive_gates'] = numbers[1]
        d['online_time.bool.layer_finish'] = numbers[2]
        
        d['online_time.yao.local_gates'] = numbers[3]
        d['online_time.yao.interactive_gates'] = numbers[4]
        d['online_time.yao.layer_finish'] = numbers[5]
        
        d['online_time.yao_rev.local_gates'] = numbers[6]
        d['online_time.yao_rev.interactive_gates'] = numbers[7]
        d['online_time.yao_rev.layer_finish'] = numbers[8]
        
        d['online_time.arith.local_gates'] = numbers[9]
        d['online_time.arith.interactive_gates'] = numbers[10]
        d['online_time.arith.layer_finish'] = numbers[11]
        
        d['online_time.splut.local_gates'] = numbers[12]
        d['online_time.splut.interactive_gates'] = numbers[13]
        d['online_time.splut.layer_finish'] = numbers[14]
        
        d['online_time.communication'] = numbers[15]
    

        # complexities
        d['complexities.boolean_sharing.ands'] = numbers[16]
        d['complexities.boolean_sharing.depth'] = numbers[18]

        d['complexities.total_vec_and'] = numbers[19]
        d['complexities.total_non_vec_and'] = numbers[20]
        d['complexities.xor_vals'] = numbers[21]
        d['complexities.gates'] = numbers[22]
        d['complexities.comb_gates'] = numbers[23]
        d['complexities.combstruct_gates'] = numbers[24]
        d['complexities.perm_gates'] = numbers[25]
        d['complexities.subset_gates'] = numbers[26]
        d['complexities.split_gates'] = numbers[27]

        d['complexities.yao.ands'] = numbers[28]
        d['complexities.yao.depth'] = numbers[29]

        d['complexities.reverse_yao.ands'] = numbers[30]
        d['complexities.reverse_yao.depth'] = numbers[31]

        d['complexities.arithmetic_sharing.muls'] = numbers[32]
        d['complexities.arithmetic_sharing.depth'] = numbers[33]

        d['complexities.sp_lut_sharing.ot_gates_total'] = numbers[34]
        d['complexities.sp_lut_sharing.depth'] = numbers[35]

        d['complexities.total_nr_of_gates'] = numbers[36]
        d['complexities.total_depth'] = numbers[37]


        # timings
        d['timings.total'] = numbers[38]
        d['timings.init'] = numbers[39]
        d['timings.circuitgen'] = numbers[40]
        d['timings.network'] = numbers[41]
        d['timings.baseots'] = numbers[42]
        d['timings.setup'] = numbers[43]
        d['timings.otextension'] = numbers[44]
        d['timings.garbling'] = numbers[45]
        d['timings.online'] = numbers[46]

        # communication
        d['communication.total.sent'] = numbers[47]
        d['communication.total.received'] = numbers[48]

        d['communication.base_ots.sent'] = numbers[49]
        d['communication.base_ots.received'] = numbers[50]

        d['communication.setup.sent'] = numbers[51]
        d['communication.setup.received'] = numbers[52]

        d['communication.otextension.sent'] = numbers[53]
        d['communication.otextension.received'] = numbers[54]

        d['communication.garbling.sent'] = numbers[55]
        d['communication.garbling.received'] = numbers[56]

        d['communication.online.sent'] = numbers[57]
        d['communication.online.received'] = numbers[58]

        # results
        d['cos_dist_ver'] = numbers[59]
        d['cos_dist_sfe'] = numbers[60]
    except:
        return None
    return d


def setup_logging(name):
    # Create a logger
    logger = logging.getLogger("pffrocd")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a handler for logging to stdout (info level)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Create a handler for logging to a file (debug level)
    file_handler = logging.FileHandler(f'log/debug_{name}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Function to execute a command on a remote host
def execute_command(host, username, command, private_key_path):
    
    # Load the private key from the specified file path
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

    # Create an SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote host using the provided credentials
    ssh.connect(hostname=host, username=username, pkey=private_key)

    # Execute the command on the remote host
    _, stdout, stderr = ssh.exec_command(command)

    # Read and decode the output of the command
    output_stdout = stdout.read().decode().strip()
    output_stderr = stderr.read().decode().strip()

    # Close the SSH connection
    ssh.close()

    # Return the output of the command
    return output_stdout, output_stderr

def send_file_to_remote_host(hostname, username, private_key_path, local_path, remote_path):
    # Create an SSH client
    client = paramiko.SSHClient()

    # Automatically add the remote host's SSH key
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Load the private key
        private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

        # Connect to the remote host using the private key for authentication
        client.connect(hostname, username=username, pkey=private_key)

        # Create an SFTP session
        sftp = client.open_sftp()

        # Upload the local file to the remote host
        sftp.put(local_path, remote_path)

        # Close the SFTP session
        sftp.close()
    finally:
        # Close the SSH client connection
        client.close()

def write_share_to_remote_file(hostname, username, private_key_path, remote_path, content: np.ndarray):
    # Load the private key from the specified file path
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

    # Create an SSH client
    client = paramiko.SSHClient()

    # Automatically add the remote host's SSH key
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote host
        client.connect(hostname, username=username, pkey=private_key)

        # Create an SFTP session
        sftp = client.open_sftp()

        with sftp.open(remote_path, 'w') as file:
            # Write the content to the file
            for i in content:
                file.write(f"{i}\n")

        # Close the SFTP session
        sftp.close()
    finally:
        # Close the SSH client connection
        client.close()


def write_embeddings_to_remote_file(hostname, username, private_key_path, remote_path, x: np.ndarray, y: np.ndarray):
    # Load the private key from the specified file path
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

    # Create an SSH client
    client = paramiko.SSHClient()

    # Automatically add the remote host's SSH key
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote host
        client.connect(hostname, username=username, pkey=private_key)

        # Create an SFTP session
        sftp = client.open_sftp()

        with sftp.open(remote_path, 'w') as file:
            # Write the content to the file
            for x_i, y_i in zip(x, y):
                file.write(f"{x_i} {y_i}\n")

        # Close the SFTP session
        sftp.close()
    finally:
        # Close the SSH client connection
        client.close()

def get_people_with_multiple_images(root_dir):
    people_folders = os.listdir(root_dir)
    people_with_multiple_images = []

    for person_folder in people_folders:
        person_path = os.path.join(root_dir, person_folder)
        if os.path.isdir(person_path):
            images = os.listdir(person_path)
            if len(images) > 1:
                people_with_multiple_images.append(person_path)

    return sorted(people_with_multiple_images)

import os

def create_shares(x: np.ndarray):
    """Create shares for the client and server from an image"""

    # generate nonces
    r = generate_nonce(x)

    # server's part is the nonces
    share1 = r

    # client's part is the nonces xored with the embedding
    share0 = fxor(x, share1)

    return share0, share1

def get_images_in_folder(folder_path):
    images = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            images.append(file_path)
    return sorted(images)


import os
import random

def get_random_images_except_person(root_dir, excluded_person, num_images):
    # Get the list of all people folders in the root directory
    people_folders = [os.path.join(root_dir, directory) for directory in os.listdir(root_dir)]

    # Remove the excluded person from the list of people folders
    people_folders.remove(excluded_person)

    # Create an empty list to store the paths to random images
    random_image_paths = []

    for _ in range(num_images):

        # sample a random person from the list
        person_folder = random.choice(people_folders)

        # Get the list of images in the person folder
        images = os.listdir(person_folder)

        # Choose a random image from the person folder
        random_image = random.choice(images)

        # Create the path of the random image
        random_image_path = os.path.join(person_folder, random_image)

        # Add the path to the list of random image paths
        random_image_paths.append(random_image_path)

    # Return the list of random image paths
    return random_image_paths


def execute_command_parallel_alternative(hosts, user, password, command1, command2):
    client = ParallelSSHClient(hosts=hosts, user=user, password=password, timeout=600)
    output = client.run_command('%s', host_args=(command1,command2))
    client.join(output)
    del client
    return output

if __name__ == "__main__":
    # host1 = "192.168.50.55"
    # host2 = "192.168.50.190"
    # hosts = [host1, host2]
    # username = "dietpi"
    # private_key_path = "/home/kamil/.ssh/id_thesis"
    # command = "ls -l"
    # # output1, output2 = execute_command_parallel(host1, username, private_key_path, command, host2, username, command)
    # send_file_to_remote_host(host1, username, private_key_path, "/home/kamil/Documents/uni/thesis/pffrocd/lfw/George_W_Bush/George_W_Bush_0001.jpg", "/home/dietpi/testimg.jpg")
    # print(config.sections())

    # share0, share1 = create_shares("/home/kamil/Documents/uni/thesis/pffrocd/lfw/George_W_Bush/George_W_Bush_0001.jpg")
    # print(share0, share1)

    # client = ParallelSSHClient(hosts=['192.168.1.66', '192.168.1.95'], user='dietpi', password="kamil123")
    # output = client.run_command('ls -l')
    # for host_out in output:
    #     for line in host_out.stdout:
    #         print(line)
    with open("/home/kamil/Documents/uni/thesis/pffrocd/sample_aby_output.txt", "r") as f:
        output = f.read()
    d = parse_aby_output(output)
    print(d.keys())