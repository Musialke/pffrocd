#!/usr/bin/env python3
"""Script for testing ram and energy usage"""
import pffrocd

# run the sleep command for 3 seconds on a remote host and gather ram and energy usage

stdout, stderr = pffrocd.execute_command("192.168.5.112", "kamil", "nice -n -20 /usr/bin/time -v sleep 3", "/home/kamil/.ssh/id_thesis")

print(pffrocd.parse_usr_bin_time_output(stderr).keys())