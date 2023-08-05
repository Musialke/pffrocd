"""
The main testing script that connects to the client and server, runs the tests and collects data
"""
import pffrocd
import logging
import configparser

"""Parse config file"""
config = configparser.ConfigParser()
config.read('config.ini')

client_ip = config.get('client', 'ip_address')
client_username = config.get('client', 'username')
client_key = config.get('client', 'private_key_path')
client_exec_path = config.get('client', 'executable_path')
client_exec_name = config.get('client', 'executable_name')

server_ip = config.get('server', 'ip_address')
server_username = config.get('server', 'username')
server_key = config.get('server', 'private_ssh_key_path')
server_exec_path = config.get('server', 'executable_path')
server_exec_name = config.get('server', 'executable_name')

test_mode = config.getboolean('misc', 'test_mode')


def run_test():
    pffrocd.setup_logging()
    config = pffrocd.parse_config()

    # print all config options to the debug log
    logging.debug(f"pffrocd config: {pffrocd.get_config_in_printing_format(config)}")

    # todo: get the list of people that have more than one image
    people = pffrocd.get_people_with_multiple_images(root_dir='lfw')

    if test_mode:
        # if debugging, only use one person
        people = people[:1]
        logging.info("RUNNING IN TEST MODE: only using one person")


    # run for all the people with multiple images
    for count, person in enumerate(people):
        # get all images from person
        print(f"{person=}")
        imgs = pffrocd.get_images_in_folder(person)
        logging.info(f"Currently running for {person} ({count}/{len(people)})")
        logging.debug(f"Found {len(imgs)} images for {person}")

        # set the first image as the 'reference' image (registered at the service provider) and remove it from the list of images
        ref_img = imgs[0]
        imgs = imgs[1:]
        logging.debug(f"setting image as reference image: {ref_img}")

        # get as many images of other people as there are of that person
        other_imgs = pffrocd.get_random_images_except_person(root_dir='lfw', excluded_person=person, num_images=len(imgs))

        # join the two list of images together
        imgs = imgs + other_imgs

        # create shares of the reference image and send them to the client and server
        share0, share1 = pffrocd.create_shares(ref_img)

        # write the shares to the server and client
        pffrocd.write_to_remote_file(client_ip, client_username, client_key, f"{client_exec_path}/share0.txt", share0)
        pffrocd.write_to_remote_file(server_ip, server_username, server_key, f"{server_exec_path}/share1.txt", share1)

        # run the test for each image
        for img in imgs:
            logging.debug(f"Running test for {img}")

            # send the image to the server
            pffrocd.write_to_remote_file(server_ip, server_username, server_key, f"{server_exec_path}/{img}", img)

            logging.debug("Image sent to server")

            # run the face embedding extraction script on the server
            pffrocd.execute_command(server_ip, server_username, server_key, command=f"python3 {server_exec_path}/extract_embedding.py -f {img}")

            logging.debug("Embedding extracted by the server")
            logging.debug("Running sfe...")

            # run the sfe on both client and server in parallel
            pffrocd.execute_command_parallel(host1=client_ip, username1=client_ip, command1=f"{client_exec_path}/{client_exec_name} -r 0", host2=server_ip, username2=server_username, command2=f"{server_exec_path}/{server_exec_name} -r 1")

            logging.debug("sfe done")

            # todo: save all results and timing data

            # todo: rerun the routine with powertop to gather energy consumption data


if __name__ == "__main__":
    run_test()