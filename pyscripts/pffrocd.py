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
import sys
import json
from pssh.clients import ParallelSSHClient
from pssh.config import HostConfig
import pandas as pd
from deepface.commons.distance import findThreshold
import csv

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
    'share_bytes_setup_phase_total',
    'share_bytes_setup_phase_client',
    'share_bytes_setup_phase_server',
    'share_bytes_online_phase_client',
    'server.Command being timed',
    'server.User time (seconds)',
    'server.System time (seconds)',
    'server.Percent of CPU this job got',
    'server.Elapsed (wall clock) time (h:mm:ss or m:ss)',
    'server.Average shared text size (kbytes)',
    'server.Average unshared data size (kbytes)',
    'server.Average stack size (kbytes)',
    'server.Average total size (kbytes)',
    'server.Maximum resident set size (kbytes)',
    'server.Average resident set size (kbytes)',
    'server.Major (requiring I/O) page faults',
    'server.Minor (reclaiming a frame) page faults',
    'server.Voluntary context switches',
    'server.Involuntary context switches',
    'server.Swaps',
    'server.File system inputs',
    'server.File system outputs',
    'server.Socket messages sent',
    'server.Socket messages received',
    'server.Signals delivered',
    'server.Page size (bytes)',
    'server.Exit status',
    'client.Command being timed',
    'client.User time (seconds)',
    'client.System time (seconds)',
    'client.Percent of CPU this job got',
    'client.Elapsed (wall clock) time (h:mm:ss or m:ss)',
    'client.Average shared text size (kbytes)',
    'client.Average unshared data size (kbytes)',
    'client.Average stack size (kbytes)',
    'client.Average total size (kbytes)',
    'client.Maximum resident set size (kbytes)',
    'client.Average resident set size (kbytes)',
    'client.Major (requiring I/O) page faults',
    'client.Minor (reclaiming a frame) page faults',
    'client.Voluntary context switches',
    'client.Involuntary context switches',
    'client.Swaps',
    'client.File system inputs',
    'client.File system outputs',
    'client.Socket messages sent',
    'client.Socket messages received',
    'client.Signals delivered',
    'client.Page size (bytes)',
    'client.Exit status',
    'energy_client',
    'energy_server',
    'server.hardware.aes.performance',
    'server.hardware.rtt',
    'server.hardware.throughput',
    'server.online_time.bool.local_gates',
    'server.online_time.bool.interactive_gates',
    'server.online_time.bool.layer_finish',
    'server.online_time.yao.local_gates',
    'server.online_time.yao.interactive_gates',
    'server.online_time.yao.layer_finish',
    'server.online_time.yao_rev.local_gates',
    'server.online_time.yao_rev.interactive_gates',
    'server.online_time.yao_rev.layer_finish',
    'server.online_time.arith.local_gates',
    'server.online_time.arith.interactive_gates',
    'server.online_time.arith.layer_finish',
    'server.online_time.splut.local_gates',
    'server.online_time.splut.interactive_gates',
    'server.online_time.splut.layer_finish',
    'server.online_time.communication',
    'server.complexities.boolean_sharing.ands',
    'server.complexities.boolean_sharing.depth',
    'server.complexities.total_vec_and',
    'server.complexities.total_non_vec_and',
    'server.complexities.xor_vals',
    'server.complexities.gates',
    'server.complexities.comb_gates',
    'server.complexities.combstruct_gates',
    'server.complexities.perm_gates',
    'server.complexities.subset_gates',
    'server.complexities.split_gates',
    'server.complexities.yao.ands',
    'server.complexities.yao.depth',
    'server.complexities.reverse_yao.ands',
    'server.complexities.reverse_yao.depth',
    'server.complexities.arithmetic_sharing.muls',
    'server.complexities.arithmetic_sharing.depth',
    'server.complexities.sp_lut_sharing.ot_gates_total',
    'server.complexities.sp_lut_sharing.depth',
    'server.complexities.total_nr_of_gates',
    'server.complexities.total_depth',
    'server.timings.total',
    'server.timings.init',
    'server.timings.circuitgen',
    'server.timings.network',
    'server.timings.baseots',
    'server.timings.setup',
    'server.timings.otextension',
    'server.timings.garbling',
    'server.timings.online',
    'server.communication.total.sent',
    'server.communication.total.received',
    'server.communication.base_ots.sent',
    'server.communication.base_ots.received',
    'server.communication.setup.sent',
    'server.communication.setup.received',
    'server.communication.otextension.sent',
    'server.communication.otextension.received',
    'server.communication.garbling.sent',
    'server.communication.garbling.received',
    'server.communication.online.sent',
    'server.communication.online.received',
    'server.cos_dist_ver',
    'server.cos_dist_sfe',
    'client.hardware.aes.performance',
    'client.hardware.rtt',
    'client.hardware.throughput',
    'client.online_time.bool.local_gates',
    'client.online_time.bool.interactive_gates',
    'client.online_time.bool.layer_finish',
    'client.online_time.yao.local_gates',
    'client.online_time.yao.interactive_gates',
    'client.online_time.yao.layer_finish',
    'client.online_time.yao_rev.local_gates',
    'client.online_time.yao_rev.interactive_gates',
    'client.online_time.yao_rev.layer_finish',
    'client.online_time.arith.local_gates',
    'client.online_time.arith.interactive_gates',
    'client.online_time.arith.layer_finish',
    'client.online_time.splut.local_gates',
    'client.online_time.splut.interactive_gates',
    'client.online_time.splut.layer_finish',
    'client.online_time.communication',
    'client.complexities.boolean_sharing.ands',
    'client.complexities.boolean_sharing.depth',
    'client.complexities.total_vec_and',
    'client.complexities.total_non_vec_and',
    'client.complexities.xor_vals',
    'client.complexities.gates',
    'client.complexities.comb_gates',
    'client.complexities.combstruct_gates',
    'client.complexities.perm_gates',
    'client.complexities.subset_gates',
    'client.complexities.split_gates',
    'client.complexities.yao.ands',
    'client.complexities.yao.depth',
    'client.complexities.reverse_yao.ands',
    'client.complexities.reverse_yao.depth',
    'client.complexities.arithmetic_sharing.muls',
    'client.complexities.arithmetic_sharing.depth',
    'client.complexities.sp_lut_sharing.ot_gates_total',
    'client.complexities.sp_lut_sharing.depth',
    'client.complexities.total_nr_of_gates',
    'client.complexities.total_depth',
    'client.timings.total',
    'client.timings.init',
    'client.timings.circuitgen',
    'client.timings.network',
    'client.timings.baseots',
    'client.timings.setup',
    'client.timings.otextension',
    'client.timings.garbling',
    'client.timings.online',
    'client.communication.total.sent',
    'client.communication.total.received',
    'client.communication.base_ots.sent',
    'client.communication.base_ots.received',
    'client.communication.setup.sent',
    'client.communication.setup.received',
    'client.communication.otextension.sent',
    'client.communication.otextension.received',
    'client.communication.garbling.sent',
    'client.communication.garbling.received',
    'client.communication.online.sent',
    'client.communication.online.received',
    'client.cos_dist_ver',
    'client.cos_dist_sfe'
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

def run_sfe_improved(x,y, y_0=None, y_1=None, x_0=None, x_1=None, scalar_x0=None, scalar_x1=None, scalar_y0=None, scalar_y1=None):
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
        with open(f"{EXECUTABLE_PATH}/share0prime.txt", 'w') as f:
            for i in x_0:
                f.write(f"{i}\n")
        with open(f"{EXECUTABLE_PATH}/share1prime.txt", 'w') as f:
            for i in x_1:
                f.write(f"{i}\n")
        # scalar
        if scalar_x0 is not None and scalar_x1 is not None:
            with open(f"{EXECUTABLE_PATH}/share0scalar_x.txt", 'w') as f:
                for i in scalar_x0:
                    f.write(f"{i}\n")
            with open(f"{EXECUTABLE_PATH}/share1scalar_x.txt", 'w') as f:
                for i in scalar_x1:
                    f.write(f"{i}\n")

            with open(f"{EXECUTABLE_PATH}/share0scalar_y.txt", 'w') as f:
                for i in scalar_y0:
                    f.write(f"{i}\n")
            with open(f"{EXECUTABLE_PATH}/share1scalar_y.txt", 'w') as f:
                for i in scalar_y1:
                    f.write(f"{i}\n")
        

    CMD = f"./{EXECUTABLE_NAME} -r 0 -f {INPUT_FILE_NAME} -o {OUTPUT_FILE_NAME} & (./{EXECUTABLE_NAME} -r 1 -f {INPUT_FILE_NAME} -o {OUTPUT_FILE_NAME} 2>&1 > /dev/null)"
    output = subprocess.run(CMD, shell=True, capture_output=True, text=True, cwd=EXECUTABLE_PATH)
    assert (output.returncode == 0), f"{output.stdout=}, {output.stderr=}" # make sure the process executed successfully
    return output


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
    CMD = f"./{EXECUTABLE_NAME} -r 0 -f {INPUT_FILE_NAME} -o {OUTPUT_FILE_NAME} & (./{EXECUTABLE_NAME} -r 1 -f {INPUT_FILE_NAME} -o {OUTPUT_FILE_NAME} 2>&1 > /dev/null)"
    output = subprocess.run(CMD, shell=True, capture_output=True, text=True, cwd=EXECUTABLE_PATH)
    assert (output.returncode == 0), f"{output.stdout=}, {output.stderr=}" # make sure the process executed successfully
    return output

def get_embedding(imagepath, dtype):
    return np.array(DeepFace.represent(img_path = imagepath, model_name="SFace", enforce_detection=False)[0]["embedding"], dtype=dtype)

def parse_powertop_csv(filepath):
    """
    Parse a PowerTop CSV file and extract a DataFrame containing the software power consumers.

    Args:
        filepath (str): The path to the PowerTop CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the software power consumers.
    """
    with open(filepath, 'r') as csv_file:
        rows = list(csv.reader(csv_file, delimiter=';'))
        begin = rows.index([' *  *  *   Overview of Software Power Consumers   *  *  *'])
        end = rows.index([' *  *  *   Device Power Report   *  *  *'])
        headers = rows[begin+2]
        return pd.DataFrame(rows[begin+3:end-2], columns=headers)

def get_energy_consumption(hostname, username, private_key_path, remote_path, running_time) -> pd.DataFrame:
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

        with sftp.open(remote_path, 'r') as file:
            rows = list(csv.reader(file, delimiter=';'))
            # get the software power consumers overview
            begin = rows.index([' *  *  *   Overview of Software Power Consumers   *  *  *'])
            end = rows.index([' *  *  *   Device Power Report   *  *  *'])
            headers = rows[begin+2]
            df = pd.DataFrame(rows[begin+3:end-2], columns=headers)

            # get the energy consumption readings of cos_dist
            # fixme: hardcoded exec name, should be read from config but powertop doesn't print full name
            mask = df['Description'].str.contains('cos_dist')
            mask.fillna(False, inplace=True)
            energy_series = df[mask]['PW Estimate']

            # combine readings from all threads and compute total energy (E = P * T)
            total_watts = 0
            for row in energy_series:
                value, unit = row.split()
                value = float(value)
                if unit == 'mW':
                    value*=0.001
                elif unit == 'uW':
                    value*=0.000001
                elif unit == 'W':
                    pass
                else:
                    return None
                total_watts += value
            total_joules = total_watts * running_time


        # Close the SFTP session
        sftp.close()
    finally:
        # Close the SSH client connection
        client.close()
        return energy_series, total_joules

def get_two_random_embeddings(same_person):
    # print(os.getcwd())
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

fxor64 = lambda x,y:(x.view("int64")^y.view("int64")).view("float64")
fxor32 = lambda x,y:(x.view("int32")^y.view("int32")).view("float32")
def fxor(x,y, dtype):
    if dtype == np.float64:
        return fxor64(x,y)
    elif dtype == np.float32:
        return fxor32(x,y)
    else:
        raise Exception("Invalid dtype")
 
def fminus(x,y, dtype):
    if dtype == np.float64:
        return x - y
    elif dtype == np.float32:
        return np.float32(x - y)
    else:
        raise Exception("Invalid dtype")

def generate_nonce(a, dtype, quantized=False):
    """Generates random float nonces given a list of floats of size 128 (the face emedding)
    Checks for nan values after xoring, if that happens then it generates the nonces again
    """

    #TODO: Add a parameter such that it is possible to switch between xor and minus...
    n = np.zeros(128)
    for i in range(len(a)):
        if quantized:
            x = np.random.randint(-128,128)
        else:
            x = np.random.uniform(-3,3)
        if dtype == np.float32:
            x = np.float32(x)
        elif dtype == np.float64:
            x = np.float64(x)
        else:
            raise Exception("Invalid dtype")
        if quantized:
            n_i = fminus(a[i], x, dtype)
        else:
            n_i = fxor(a[i], x, dtype)  

        while np.isnan(n_i):
            if quantized:
                x = np.random.randint(-128,128)
            else:
                x = np.random.uniform(-3,3)
            if dtype == np.float32:
                x = np.float32(x)
            elif dtype == np.float64:
                x = np.float64(x)
            else:
                raise Exception("Invalid dtype")
            if quantized:
                n_i = fminus(a[i], x, dtype)
            else:
                n_i = fxor(a[i], x, dtype)  
        n[i] = n_i
    return n.astype(dtype)

def parse_usr_bin_time_output(output):
    """Parses the benchmark output of usr/bin/time and returns stats of interest in a dictionary"""
    metrics = {}
    lines = output.strip().split('\t')

    for line in lines:
        try:
            key, value = line.split(': ')
        except:
            return None
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
        # hardware
        d['hardware.aes.performance'] = numbers[0]
        d['hardware.rtt'] = numbers[1]
        d['hardware.throughput'] = numbers[2]

        # online_time
        d['online_time.bool.local_gates'] = numbers[3]
        d['online_time.bool.interactive_gates'] = numbers[4]
        d['online_time.bool.layer_finish'] = numbers[5]

        d['online_time.yao.local_gates'] = numbers[6]
        d['online_time.yao.interactive_gates'] = numbers[7]
        d['online_time.yao.layer_finish'] = numbers[8]

        d['online_time.yao_rev.local_gates'] = numbers[9]
        d['online_time.yao_rev.interactive_gates'] = numbers[10]
        d['online_time.yao_rev.layer_finish'] = numbers[11]

        d['online_time.arith.local_gates'] = numbers[12]
        d['online_time.arith.interactive_gates'] = numbers[13]
        d['online_time.arith.layer_finish'] = numbers[14]

        d['online_time.splut.local_gates'] = numbers[15]
        d['online_time.splut.interactive_gates'] = numbers[16]
        d['online_time.splut.layer_finish'] = numbers[17]

        d['online_time.communication'] = numbers[18]


        # complexities
        d['complexities.boolean_sharing.ands'] = numbers[19]
        d['complexities.boolean_sharing.depth'] = numbers[21]

        d['complexities.total_vec_and'] = numbers[22]
        d['complexities.total_non_vec_and'] = numbers[23]
        d['complexities.xor_vals'] = numbers[24]
        d['complexities.gates'] = numbers[25]
        d['complexities.comb_gates'] = numbers[26]
        d['complexities.combstruct_gates'] = numbers[27]
        d['complexities.perm_gates'] = numbers[28]
        d['complexities.subset_gates'] = numbers[29]
        d['complexities.split_gates'] = numbers[30]

        d['complexities.yao.ands'] = numbers[31]
        d['complexities.yao.depth'] = numbers[32]

        d['complexities.reverse_yao.ands'] = numbers[33]
        d['complexities.reverse_yao.depth'] = numbers[34]

        d['complexities.arithmetic_sharing.muls'] = numbers[35]
        d['complexities.arithmetic_sharing.depth'] = numbers[36]

        d['complexities.sp_lut_sharing.ot_gates_total'] = numbers[37]
        d['complexities.sp_lut_sharing.depth'] = numbers[38]

        d['complexities.total_nr_of_gates'] = numbers[39]
        d['complexities.total_depth'] = numbers[40]


        # timings
        d['timings.total'] = numbers[41]
        d['timings.init'] = numbers[42]
        d['timings.circuitgen'] = numbers[43]
        d['timings.network'] = numbers[44]
        d['timings.baseots'] = numbers[45]
        d['timings.setup'] = numbers[46]
        d['timings.otextension'] = numbers[47]
        d['timings.garbling'] = numbers[48]
        d['timings.online'] = numbers[49]

        # communication
        d['communication.total.sent'] = numbers[50]
        d['communication.total.received'] = numbers[51]

        d['communication.base_ots.sent'] = numbers[52]
        d['communication.base_ots.received'] = numbers[53]

        d['communication.setup.sent'] = numbers[54]
        d['communication.setup.received'] = numbers[55]

        d['communication.otextension.sent'] = numbers[56]
        d['communication.otextension.received'] = numbers[57]

        d['communication.garbling.sent'] = numbers[58]
        d['communication.garbling.received'] = numbers[59]

        d['communication.online.sent'] = numbers[60]
        d['communication.online.received'] = numbers[61]

        # results
        d['cos_dist_ver'] = numbers[62]
        d['cos_dist_sfe'] = numbers[63]
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
            total_bytes = 0
            for i in content:
                line = f"{i}\n"
                total_bytes += sys.getsizeof(line)
                file.write(line)

        # Close the SFTP session
        sftp.close()
    finally:
        # Close the SSH client connection
        client.close()

    return total_bytes


def write_embeddings_to_remote_file(hostname, username, private_key_path, remote_path, x: np.ndarray, y: np.ndarray):
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

def get_bandwidth(hostname1, hostname2, username1, username2, password1, password2, private_key_path, remote_path, current_datetime):
    '''execute iperf3 command on a list of hosts'''
    output = execute_command_parallel_alternative([hostname1, hostname2], username1, username2, password1, password2,  f'iperf3 -J -s -1 --logfile iperf3_{current_datetime}.log', f'sleep 3 && iperf3 -J -c {hostname1}', timeout=20)
    for host_output in output:
        stdout = list(host_output.stdout)
        stderr = list(host_output.stderr)
        # if host_output.exit_code != 0:
        #     return f"f'iperf3 failed on {host_output.host}. stdout: {stdout}, stderr: {stderr}"
    '''read the logfile into a dictionary'''
    
    # Load the private key from the specified file path
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

    # Create an SSH client
    client = paramiko.SSHClient()

    # Automatically add the remote host's SSH key
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote host
        client.connect(hostname1, username=username1, pkey=private_key)

        # Create an SFTP session
        sftp = client.open_sftp()

        with sftp.open(f'iperf3_{current_datetime}.log', 'r') as file:
            # Read the content of the file
            content = file.readlines()
            # print(f"{content=}")
            # load the content into a dictionary
            d = json.loads(''.join(content))
            # print(f"{d=}")
            bandwidth = float(d['end']['sum_received']['bits_per_second']) / 1e6

        # remove the file on the host
        sftp.remove(f'iperf3_{current_datetime}.log')

        # Close the SFTP session
        sftp.close()
    finally:
        # Close the SSH client connection
        client.close()
        return bandwidth


def create_shares(x: np.ndarray, dtype, quantized=False):
    """Create shares for the client and server from an image"""

    # generate nonces
    r = generate_nonce(x, dtype, quantized)

    # server's part is the nonces
    share1 = r

    # client's part is the nonces xored with the embedding
    #share0 = fxor(x, share1, dtype)
    if quantized:
        share0 = fminus(x, share1, dtype)
    else:
        share0 = fxor(x, share1, dtype)

    return share0.astype(dtype), share1.astype(dtype)

def create_share(x: float):
    
    #nonce = fxor(x, np.random.uniform(-3,3), np.float32) 
    nonce = x ^ np.random.uniform(-3,3)
    print(nonce)

    share1 = nonce

    share0 = x ^ share1

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


def execute_command_parallel_alternative(hosts, username1, username2, password1, password2, command1, command2, timeout=600):
    host_config = [
            HostConfig(port=22, user=username1,
                    password=password1),
            HostConfig(port=22, user=username2,
                    password=password2),
            ]
    client = ParallelSSHClient(hosts=hosts, host_config=host_config, timeout=timeout)
    output = client.run_command('%s', host_args=(command1,command2), sudo=True)
    for host_out in output:
        if host_out.host == hosts[0]:
            host_out.stdin.write(f'{password1}\n')
            host_out.stdin.flush()
        elif host_out.host == hosts[1]:
            host_out.stdin.write(f'{password2}\n')
            host_out.stdin.flush()
        else:
            raise Exception(f"Unexpected host: {host_out.host}")
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
    # print(d.keys())