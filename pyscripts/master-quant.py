"""
The main testing script that connects to the client and server, runs the tests and collects data
"""
import datetime
import os
import pffrocd
import configparser
import time
import pandas as pd
import numpy as np
import quantization as qt

sec_lvl = None
mt_alg = None


"""Parse config file"""
config = configparser.ConfigParser()
config.read('config.ini')

# override config if parameters already provided
def orverride_config(sec_lvl, mt_alg):
        config.set('misc', 'security_level', str(sec_lvl))
        config.set('misc', 'mt_algorithm', str(mt_alg))

client_ip = config.get('client', 'ip_address')
client_username = config.get('client', 'username')
client_password = config.get('client', 'password')
client_key = config.get('client', 'private_ssh_key_path')
client_exec_path = config.get('client', 'executable_path')
client_exec_name = config.get('client', 'executable_name')
client_pffrocd_path = config.get('client', 'pffrocd_path')

server_ip = config.get('server', 'ip_address')
server_username = config.get('server', 'username')
server_password = config.get('server', 'password')
server_key = config.get('server', 'private_ssh_key_path')
server_exec_path = config.get('server', 'executable_path')
server_exec_name = config.get('server', 'executable_name')
server_pffrocd_path = config.get('server', 'pffrocd_path')

master_key_path = config.get('master', 'private_ssh_key_path')

nr_of_people = config.getint('misc', 'nr_of_people')
niceness = config.getint('misc', 'niceness')
starting_person = config.getint('misc', 'starting_person')
bit_length = config.getint('misc', 'bit_length')
gather_energy_data = config.getint('misc', 'gather_energy_data')

if bit_length == 64:
    NUMPY_DTYPE = np.float64
elif bit_length == 32:
    NUMPY_DTYPE = np.float32
elif bit_length == 16:
    NUMPY_DTYPE = np.float16
else:
    raise Exception("Invalid bit length")

# client_exec_name += f"_{bit_length}"
# server_exec_name += f"_{bit_length}"

def run_test():
    output_path = 'dfs/2024-06-19_19-25-59.csv'

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    sec_lvl = config.getint('misc', 'security_level')
    mt_alg = config.getint('misc', 'mt_algorithm')
    logger = pffrocd.setup_logging(current_datetime)

    # print all config options to the debug log
    logger.debug(f"pffrocd config: {pffrocd.get_config_in_printing_format(config)}")

    # get the bandwidth and log it
    bandwidth = pffrocd.get_bandwidth(server_ip, client_ip, server_username, client_username, server_password, client_password, master_key_path, server_pffrocd_path, current_datetime)
    logger.info(f"Initial (tested with iperf3) bandwidth: {bandwidth:.2f} Mbits/sec")

    # get the list of people that have more than one image
    people = pffrocd.get_people_with_multiple_images(root_dir='lfw')
    people = people[starting_person:nr_of_people]
    logger.info(f"RUNNING for {len(people)} people")

    # list to store data (later saved as a dataframe) see https://stackoverflow.com/a/56746204
    # data = []

    # run for all the people with multiple images
    for count_person, person in enumerate(people):
        # get all images from person
        imgs = pffrocd.get_images_in_folder(person)
        logger.info(f"Currently running for {person} ({count_person+1}/{len(people)})")
        logger.debug(f"Found {len(imgs)} images for {person}")

        # set the first image as the 'reference' image (registered at the service provider) and remove it from the list of images
        ref_img = imgs[0]
        imgs = imgs[1:]
        logger.debug(f"setting image as reference image: {ref_img}")

        # get as many images of other people as there are of that person
        other_imgs = pffrocd.get_random_images_except_person(root_dir='lfw', excluded_person=person, num_images=len(imgs))

        # join the two list of images together
        imgs = imgs + other_imgs

        # create shares of the reference image
        ref_img_embedding = pffrocd.get_embedding(ref_img, dtype=NUMPY_DTYPE)
        
        # NORMALIZE THE FACE EMBEDDING SO THE SHARES ARE NORMALIZED TO
        ref_img_embedding = ref_img_embedding / np.linalg.norm(ref_img_embedding)
        ref_img_embedding = qt.scalar_quantisation_percentile(ref_img_embedding)
        
        share0, share1 = pffrocd.create_shares(np.array(ref_img_embedding, dtype=NUMPY_DTYPE), dtype=NUMPY_DTYPE, quantized=True)
        share0 = np.array(share0, dtype=np.int32)
        share1 = np.array(share1, dtype=np.int32)
        norm_ref_img_embedding = [np.linalg.norm(ref_img_embedding)]
        share0scalar, share1scalar = pffrocd.create_shares(np.array(norm_ref_img_embedding, dtype=NUMPY_DTYPE), dtype=NUMPY_DTYPE, quantized=False)
        # preppring scalars
        share0scalar = np.array([share0scalar[0]], dtype=np.float32)
        share1scalar = np.array([share1scalar[0]], dtype=np.float32)
        # write the shares to the server and client
        # I feel like an extra comment here is necessary. The share goes to the client, but since the client has role 1 it is written to share1.txt, such that in the cpp file we can do: share{role}.txt
        # This made me confused for a while, so I think it is good to clarify this here
        shareBytesSetupPhaseTotal = 0
        shareBytesSetupPhaseServer = 0
        shareBytesSetupPhaseClient = 0       
        shareBytesSetupPhaseClient += pffrocd.write_share_to_remote_file(client_ip, client_username, master_key_path, f"{client_exec_path}/share1.txt", share0)
        shareBytesSetupPhaseServer += pffrocd.write_share_to_remote_file(server_ip, server_username, master_key_path, f"{server_exec_path}/share0.txt", share1)
        shareBytesSetupPhaseClient += pffrocd.write_share_to_remote_file(client_ip, client_username, master_key_path, f"{client_exec_path}/share1scalar_x.txt", share0scalar)
        shareBytesSetupPhaseServer += pffrocd.write_share_to_remote_file(server_ip, server_username, master_key_path, f"{server_exec_path}/share0scalar_x.txt", share1scalar)
        shareBytesSetupPhaseTotal = shareBytesSetupPhaseClient + shareBytesSetupPhaseServer
        # run the test for each image
        for count_img,img in enumerate(imgs):
            # if df['other_img'] already contains img, skip this iteration
            df = pd.read_csv(output_path)
            if img in df['other_img'].values:
                logger.info(f"Skipping image {img} as it is already in the dataframe")
                continue

            logger.info(f"Running test for {img}")

            # run the face embedding extraction script on the server
            stdout, stderr = pffrocd.execute_command(server_ip, server_username, f"{server_pffrocd_path}/env/bin/python {server_pffrocd_path}/pyscripts/extract_embedding.py -i {server_pffrocd_path}/{img} -o {server_exec_path}/embedding.txt -n {True}", master_key_path)
            logger.debug(f"Stdout of extracting embedding: {stdout}")
            logger.debug(f"Stderr of extracting embedding: {stderr}")
            extraction_time = float(stdout)

            if stderr != '':
                logger.error("REMOTE EXECUTION OF COMMAND FAILED")
                logger.error(stderr)

            logger.info(f"Embedding extracted by the server in {extraction_time} seconds")

            # Let the server create two shares from the embeddings
            stdout, stderr = pffrocd.execute_command(server_ip, server_username, f"{server_pffrocd_path}/env/bin/python {server_pffrocd_path}/pyscripts/generate_shares.py -i {server_exec_path}/embedding.txt -b {bit_length} -o {server_exec_path}/share0prime.txt -q {True} -s {server_exec_path}/share0scalar_y.txt", master_key_path)
            # logger.info(f"Stdout of extracting embedding: {stdout}")
            # logger.info(f"Stderr of extracting embedding: {stderr}")
            # split the string in two, parts before | and after
            s = stdout.split('|')
            scalar = s[0].strip("[]").split()
            # remove last item from scalar[0]
            scalar[0] = scalar[0][:-1]
            scalar = [np.float32(i) for i in scalar]
            scalar = np.array(scalar, dtype=np.float32)
            # Remove the brackets and split the string into a list of strings
            # Convert the list of strings into a list of floats
            s[1] = s[1].strip("[]").split()
            # remove first item from s[1]
            s[1] = s[1][1:]
            s = [np.int32(i) for i in s[1]]
            # Convert the list of floats into a numpy array
            shareprime = np.array(s, dtype=np.int32)

            if stderr != '':
                logger.error("REMOTE EXECUTION OF COMMAND FAILED")
                logger.error(stderr)
            shareBytesOnlinePhaseClient = 0
            shareBytesOnlinePhaseClient += pffrocd.write_share_to_remote_file(client_ip, client_username, master_key_path, f"{client_exec_path}/share1scalar_y.txt", scalar)
            shareBytesOnlinePhaseClient += pffrocd.write_share_to_remote_file(client_ip, client_username, master_key_path, f"{client_exec_path}/share1prime.txt", shareprime)
            
            # send the files with embeddings to the client and server
            img_embedding = pffrocd.get_embedding(img, dtype=NUMPY_DTYPE)
            img_embedding = img_embedding / np.linalg.norm(img_embedding)
            img_embedding = qt.scalar_quantisation_percentile(img_embedding)
            ref_img_embedding = np.array(ref_img_embedding, dtype=np.int32)
            img_embedding = np.array(img_embedding, dtype=np.int32)

            pffrocd.write_embeddings_to_remote_file(client_ip, client_username, master_key_path, f"{client_exec_path}/embeddings.txt", img_embedding, ref_img_embedding)
            pffrocd.write_embeddings_to_remote_file(server_ip, server_username, master_key_path, f"{server_exec_path}/embeddings.txt", img_embedding, ref_img_embedding)
            
            # run the sfe on both client and server in parallel
            logger.info("Running sfe...")
            # stdout1, stderr1, stdout2, stderr2 = pffrocd.execute_command_parallel(host1=client_ip, username1=client_username, command1=f"{client_exec_path}/{client_exec_name} -r 1 -a {server_ip} -f {client_exec_path}/embeddings.txt", host2=server_ip, username2=server_username, command2=f"{server_exec_path}/{server_exec_name} -r 0 -a {server_ip} -f {server_exec_path}/embeddings.txt", private_key_path1=client_key, private_key_path2=server_key)
            # logger.debug("sfe done")
            # logger.debug(f"{stdout1=}")
            # logger.debug(f"{stderr1=}")
            # logger.debug(f"{stdout2=}")
            # logger.debug(f"{stderr2=}")

            command1 = f"cd {client_exec_path} ; nice -n {niceness} /usr/bin/time -v {client_exec_path}/{client_exec_name} -r 1 -a {server_ip} -f {client_exec_path}/embeddings.txt -o {client_pffrocd_path} -s {sec_lvl} -x {mt_alg}"
            command2 = f"cd {server_exec_path} ; nice -n {niceness} /usr/bin/time -v {server_exec_path}/{server_exec_name} -r 0 -a {server_ip} -f {server_exec_path}/embeddings.txt -o {client_pffrocd_path} -s {sec_lvl} -x {mt_alg}"
            sfe_start_time  = time.time()
            output = pffrocd.execute_command_parallel_alternative([client_ip, server_ip], client_username, server_username, client_password, server_password, command1, command2, timeout=300)
            sfe_time = time.time() - sfe_start_time
            logger.info(f"Finished! Total sfe time: {sfe_time} seconds ({count_img+1}/{len(imgs)})")
            server_sfe_output = ''
            server_sfe_error = ''
            client_sfe_output = ''
            client_sfe_error = ''
            for host_output in output:
                hostname = host_output.host
                stdout = list(host_output.stdout)
                stderr = list(host_output.stderr)
                logger.debug("Host %s: exit code %s, output %s, error %s" % (
                    hostname, host_output.exit_code, stdout, stderr))
                if hostname == server_ip:
                    server_sfe_output = ''.join(stdout)
                    server_sfe_error = ''.join(stderr)
                if hostname == client_ip:
                    client_sfe_output = ''.join(stdout)
                    client_sfe_error = ''.join(stderr)
            server_ram_usage = pffrocd.parse_usr_bin_time_output(server_sfe_error)
            client_ram_usage = pffrocd.parse_usr_bin_time_output(client_sfe_error)
            logger.debug(f"Parsed server ram usage: {server_ram_usage}")
            logger.debug(f"Parsed client ram usage: {client_ram_usage}")
            logger.debug(f"Server SFE output: {server_sfe_output}")
            logger.debug(f"Client SFE output: {client_sfe_output}")
            # rerun the routine with powertop to gather energy consumption data
            if gather_energy_data:
                logger.info("Running powertop to gather energy consumption data...")
                running_time_client = float(client_ram_usage['User time (seconds)'])
                running_time_server = float(server_ram_usage['User time (seconds)'])
                powertop_command = f"sudo powertop --csv=powertop_{current_datetime}.csv -t {sfe_time + 1}"
                output = pffrocd.execute_command_parallel_alternative([client_ip, server_ip], client_username, server_username, client_password, server_password, f"{command1} & {powertop_command}", f"{command2} & {powertop_command}", timeout=600)
                # get the powertop files from hosts and parse them and save in the dataframe
                all_values, energy_client = pffrocd.get_energy_consumption(client_ip, client_username, master_key_path, f"{client_exec_path}/powertop_{current_datetime}.csv", running_time_client + 1)
                #logger.info(f"All values from powertop for client: {all_values}")
                #logger.info(f"Energy client: {energy_client}")
                all_values, energy_server = pffrocd.get_energy_consumption(server_ip, server_username, master_key_path, f"{server_exec_path}/powertop_{current_datetime}.csv", running_time_server + 1)
                #logger.info(f"All values from powertop for server: {all_values}")
                #logger.debug(f"Energy server: {energy_server}")
            else:
                energy_client = 'not measured'
                energy_server = 'not measured'


            # save all results and timing data
            server_parsed_sfe_output = pffrocd.parse_aby_output(server_sfe_output)
            client_parsed_sfe_output = pffrocd.parse_aby_output(client_sfe_output)
            if not server_parsed_sfe_output or not client_parsed_sfe_output:
                continue
            if not server_ram_usage or not client_ram_usage:
                continue
            logger.info(f"Server throughput (for this run, reported by ABY): {float(server_parsed_sfe_output['hardware.throughput']) * 8:.2f} Mbits/sec")
            logger.info(f"Client throughput (for this run, reported by ABY): {float(client_parsed_sfe_output['hardware.throughput']) * 8:.2f} Mbits/sec")
            logger.info(f"Server total time: {float(server_parsed_sfe_output['timings.total']) / 1000}")
            logger.info(f"Client total time: {float(client_parsed_sfe_output['timings.total']) / 1000}")
            cos_dist_sfe = float(server_parsed_sfe_output['cos_dist_sfe'])
            result = cos_dist_sfe < pffrocd.threshold
            expected_result = ref_img.split('/')[1] == img.split('/')[1] # check if the images belong to the same person
            # ref_img_embedding = np.array(ref_img_embedding, dtype=np.uint32)
            # img_embedding = np.array(img_embedding, dtype=np.uint32)
            
            cos_dist_np = pffrocd.get_cos_dist_numpy(ref_img_embedding, img_embedding)
            logger.info(f"cosine distance between embeddings: {cos_dist_np}")
            server_list_of_sfe_values = list(server_parsed_sfe_output.values())
            client_list_of_sfe_values = list(client_parsed_sfe_output.values())
            logger.debug(f"{server_parsed_sfe_output=}")
            logger.debug(f"{server_list_of_sfe_values=}")
            logger.debug(f"{client_parsed_sfe_output=}")
            logger.debug(f"{client_list_of_sfe_values=}")
            # if ram_usage asked for password for sudo delete that entry
            logger.debug(f"Checking if the key <[sudo] password for {server_username}> exists")
            server_ram_usage.pop(f'[sudo] password for {server_username}', None)
            client_ram_usage.pop(f'[sudo] password for {client_username}', None)
            server_list_of_ram_values = list(server_ram_usage.values()) # [1:] # remove the first element, asking for sudo
            client_list_of_ram_values = list(client_ram_usage.values())
            logger.debug(f"{server_ram_usage=}")
            logger.debug(f"{client_ram_usage=}")
            logger.debug(f"{server_list_of_ram_values=}")
            logger.debug(f"{client_list_of_ram_values=}")
            to_be_appended = [ref_img, img, result, expected_result, cos_dist_np, cos_dist_sfe, sfe_time + extraction_time, sfe_time, extraction_time, shareBytesSetupPhaseTotal, shareBytesSetupPhaseClient, shareBytesSetupPhaseServer, shareBytesOnlinePhaseClient] + server_list_of_ram_values + client_list_of_ram_values +  [energy_client, energy_server] + server_list_of_sfe_values + client_list_of_sfe_values
            logger.debug(f"{to_be_appended=}")
            logger.debug(f"{pffrocd.columns=}")
            # make and iteratively save the dataframe with results        
            df = pd.DataFrame([to_be_appended], columns=pffrocd.columns)
            #output_path = f"dfs/{current_datetime}.csv"
            # append dataframe to file, only write headers if file does not exist yet
            df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


if __name__ == "__main__":
    run_test()