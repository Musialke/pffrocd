# pffrocd
Privacy-Friendly Face Recognition On Constrained Devices


The testing flow is as follows:

![image](https://github.com/Musialke/pffrocd/assets/26610983/e0843c66-283b-4aea-b536-fe309f1481fd)

The code is also rather extensively commented. The main script is `pyscripts/master.py` and helping functions are in `pyscripts/pfforcd.py`.

## Complete guide to set up a host for tests:

1. Install required packages:

```sh
sudo apt update && sudo apt install time python3 python3-venv iperf3 g++ make cmake libgmp-dev libssl-dev libboost-all-dev ffmpeg libsm6 libxext6 git powertop -y
```

2. Generate SSH key and add as deploy key to the git repo

```sh
ssh-keygen
```

3. Clone the repo and cd into it

```sh
git clone git@github.com:Musialke/pffrocd.git
cd pffrocd
```

For SERVER and CLIENT:

4. Create the ABY build directory
```sh
mkdir ABY/build/ && cd ABY/build/
```

5. Use CMake to configure the build (example applications on by default):
```sh
cmake ..
```

6. Call `make` in the build directory. You can find the build executables and libraries in the directories `bin/` and `lib/`, respectively.
```sh
make
```

7. To be able to run a process with higher priority modify limits.conf as explained here: https://unix.stackexchange.com/a/358332

8. Calibrate Powertop to get power estimate readings

```sh
sudo powertop --calibrate
```

ADDITIONALLY FOR SERVER AND MASTER:

Since the server and master need to extract embeddings, they need the database of pictures and Python.

9. Change directory back to repo root folder and unpack the picture database:
```sh
cat lfw.tgz.parta* | tar -xzv
```

10. Create a new virtual environment, activate it and install the required packages
```sh
python3 -m venv env
. env/bin/activate
pip install -vr requirements.txt
```

11. Copy the SFace weights where deepface can find them:
```sh
mkdir -p ~/.deepface/weights/ && cp face_recognition_sface_2021dec.onnx ~/.deepface/weights/
```
ADDITIONALLY FOR MASTER

12. Rename the `config.ini.example` file to `config.ini` and modify it accordingly

13. Copy the ssh keys to server and client using ssh-copy-id

14. Run the main script in the background on the master machine
```sh
python pyscripts/master.py&
```

The logs are saved in the `log/` directory and the test results are appended to a csv file in `dfs/` after running all tests for one person.


### Possible errors and solutions:

`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`
fix:
```sh
sudo apt update && sudo apt install ffmpeg libsm6 libxext6  -y
```

`v2.error: OpenCV(4.7.0) /io/opencv/modules/dnn/src/onnx/onnx_importer.cpp:275: error: (-210:Unsupported format or combination of formats) Failed to parse ONNX model: /home/dietpi/.deepface/weights/face_recognition_sface_2021dec.onnx in function 'ONNXImporter'` 
The link to weights for SFace is missing. fix:
```sh
mkdir -p ~/.deepface/weights/ && cp face_recognition_sface_2021dec.onnx ~/.deepface/weights/
```
