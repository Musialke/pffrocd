"""
The main testing script that connects to the client and server, runs the tests and collects data
"""
import datetime
import os
import pffrocd
import configparser
import time
import pandas as pd

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
client_key = config.get('client', 'private_ssh_key_path')
client_exec_path = config.get('client', 'executable_path')
client_exec_name = config.get('client', 'executable_name')
client_pffrocd_path = config.get('client', 'pffrocd_path')

server_ip = config.get('server', 'ip_address')
server_username = config.get('server', 'username')
server_key = config.get('server', 'private_ssh_key_path')
server_exec_path = config.get('server', 'executable_path')
server_exec_name = config.get('server', 'executable_name')
server_pffrocd_path = config.get('server', 'pffrocd_path')

nr_of_people = config.getint('misc', 'nr_of_people')
niceness = config.getint('misc', 'niceness')
starting_person = config.getint('misc', 'starting_person')





def run_test():
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    sec_lvl = config.getint('misc', 'security_level')
    mt_alg = config.getint('misc', 'mt_algorithm')
    logger = pffrocd.setup_logging(current_datetime)
    nr_of_images = config.getint('misc', 'nr_of_images')

    # print all config options to the debug log
    logger.debug(f"pffrocd config: {pffrocd.get_config_in_printing_format(config)}")

    # get the list of people that have more than one image
    people = pffrocd.get_people_with_multiple_images(root_dir='lfw')
    people = people[starting_person:nr_of_people]
    logger.info(f"RUNNING for {len(people)} people")

    # list to store data (later saved as a dataframe) see https://stackoverflow.com/a/56746204
    data = []

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

        # we want to have nr_of_images images in total
        num_images_to_get = nr_of_images - len(imgs)
        if num_images_to_get > 0:
            other_imgs = pffrocd.get_random_images_except_person(root_dir='lfw', excluded_person=person, num_images=num_images_to_get)
            # join the two list of images together
            imgs = imgs + other_imgs
        else:
            # only work with nr_of_images images
            imgs = imgs[:nr_of_images]


        # create shares of the reference image
        ref_img_embedding = pffrocd.get_embedding(ref_img)
        share0, share1 = pffrocd.create_shares(ref_img_embedding)

        # write the shares to the server and client
        pffrocd.write_share_to_remote_file(client_ip, client_username, client_key, f"{client_exec_path}/share1.txt", share0)
        pffrocd.write_share_to_remote_file(server_ip, server_username, server_key, f"{server_exec_path}/share0.txt", share1)


        # run the test for all imgs at once
        logger.info(f"Running test for all images, {nr_of_images} in total")

        # run the face embedding extraction script on the server
        for i, img in enumerate(imgs):        
            # send the files with embeddings to the client and server
            img_embedding = pffrocd.get_embedding(img)
            pffrocd.write_embeddings_to_remote_file(client_ip, client_username, client_key, f"{client_exec_path}/embeddings{i}.txt", img_embedding, ref_img_embedding)
            pffrocd.write_embeddings_to_remote_file(server_ip, server_username, server_key, f"{server_exec_path}/embeddings{i}.txt", img_embedding, ref_img_embedding)
            
        # run the sfe on both client and server in parallel
        logger.info("Running sfe...")

        # todo: add the nr_of_images flag to the command

        command1 = f"cd {client_exec_path} ; nice -n {niceness} /usr/bin/time -v {client_exec_path}/{client_exec_name} -r 1 -a {server_ip} -f {client_exec_path}/embeddings -o {client_pffrocd_path} -s {sec_lvl} -x {mt_alg} -d {nr_of_images}"
        command2 = f"cd {server_exec_path} ; nice -n {niceness} /usr/bin/time -v {server_exec_path}/{server_exec_name} -r 0 -a {server_ip} -f {server_exec_path}/embeddings -o {client_pffrocd_path} -s {sec_lvl} -x {mt_alg} -d {nr_of_images}"
        sfe_start_time  = time.time()
        output = pffrocd.execute_command_parallel_alternative([client_ip, server_ip], client_username, "kamil123", command1, command2)
        sfe_time = time.time() - sfe_start_time
        logger.info(f"Finished! Total sfe time: {sfe_time} seconds")
        server_sfe_output = ''
        server_sfe_error = ''
        for host_output in output:
            hostname = host_output.host
            stdout = list(host_output.stdout)
            stderr = list(host_output.stderr)
            logger.debug("Host %s: exit code %s, output %s, error %s" % (
                hostname, host_output.exit_code, stdout, stderr))
            if hostname == server_ip:
                server_sfe_output = ''.join(stdout)
                server_sfe_error = ''.join(stderr)
        ram_usage = pffrocd.parse_usr_bin_time_output(server_sfe_error)
        logger.debug(f"Parsed ram usage: {ram_usage}")

        # todo: rerun the routine with powertop to gather energy consumption data

        # parse the outputs
        l = []
        benchmarks = server_sfe_output.split("split here")
        for b in benchmarks:
            parsed = pffrocd.parse_aby_output(b)
            if parsed:
                l.append(parsed)

        # for x in l:
        #     print(json.dumps(x, sort_keys=False, indent=4))
        df = pd.DataFrame(l)
        df.to_csv("hosts_reset.csv", mode='a', header=not os.path.exists("hosts_reset.csv"), index=False)
        logger.info("Finished!")






        # # save all results and timing data
        # parsed_sfe_output = pffrocd.parse_aby_output(server_sfe_output)
        # if not parsed_sfe_output:
        #     continue
        # if not ram_usage:
        #     continue
        # cos_dist_sfe = float(parsed_sfe_output['cos_dist_sfe'])
        # result = cos_dist_sfe < pffrocd.threshold
        # expected_result = ref_img.split('/')[1] == img.split('/')[1] # check if the images belong to the same person
        # cos_dist_np = pffrocd.get_cos_dist_numpy(ref_img_embedding, img_embedding)
        # list_of_sfe_values = list(parsed_sfe_output.values())
        # list_of_ram_values = list(ram_usage.values())
        # to_be_appended = [ref_img, img, result, expected_result, cos_dist_np, cos_dist_sfe, sfe_time, sfe_time, 0] + list_of_ram_values +  [0] + list_of_sfe_values
        # data.append(to_be_appended)

        
        # # make and iteratively save the dataframe with results        
        # df = pd.DataFrame(data, columns=pffrocd.columns)
        # output_path = f"dfs/{current_datetime}.csv"
        # # append dataframe to file, only write headers if file does not exist yet
        # df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
        # data = []


if __name__ == "__main__":
    run_test()