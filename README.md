# pffrocd
Privacy-Friendly Face Recognition On Constrained Devices

## Complete guide to set up a host for tests:

1. Install required packages:

```sh
sudo apt update && sudo apt install python3 python3.11-venv g++ make cmake libgmp-dev libssl-dev libboost-all-dev ffmpeg libsm6 libxext6 git -y
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

ADDITIONALLY FOR SERVER AND MASTER:
Since the server and master need to extract embeddings, they need the database of pictures and Python.

8. Change directory back to repo root folder and unpack the picture database:
```sh
cat lfw.tgz.parta* | tar -xzv
```

9. Create a new virtual environment, activate it and install the required packages
```sh
python3 -m venv env
. env/bin/activate
pip install -vr requirements.txt
```

10. Copy the SFace weights where deepface can find them:
```sh
mkdir -p ~/.deepface/weights/ && cp face_recognition_sface_2021dec.onnx ~/.deepface/weights/
```
ADDITIONALLY FOR MASTER

11. Rename the `config.ini.example` file to `config.ini` and modify it accordingly

12. Copy the ssh keys to server and client using ssh-copy-id

13. Run the main script in the background on the master machine
```sh
python pyscripts/master.py&
```


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
### Examples explanation:


- **cos_dist_float_nscen_simd**
  - Input: face embeddings, i.e.original floats
  - Circuit: non-scenario, so two SIMD IN gates
  - Goal: show that the circuit we have works with two unaltered embeddings without shared gates
- **cos_dist_roundfloat_nscen_simd**
  - Input: embeddings, but floats are scaled up by some factor and remainder after the decimal place is removed
  - Circuit: non-scenario, so two SIMD IN gates
  - Goal: See if the circuit works with input: float -> int -> float in ABY (all zeros after decimal)
- **cos_dist_int_nscen_simd**
  - Input: embeddings cast to int after scaling up by a factor
  - Circuit: non-scenario, SIMD, inputs as ints instead of floats
  - Goal: See if having ints as inputs instead of floats works
- **cos_dist_float_nscen_nsimd**
  - Input: embeddings, original floats
  - Circuit: non-scenario, **non-SIMD**
  - Goal: Have a non-SIMD version of the circuit, potentially easier to debug
- **cos_dist_roundfloat_nscen_nsimd**
  - Input: embeddings, but floats are scaled up by some factor and the remainder after the decimal place is removed
  - Circuit: non-scenario, **non-SIMD**
  - Goal: Have a non-SIMD circuit with inputs that can be cast to int
- **cos_randint_nscen_nsimd**
  - Input: vectors of random **positive** integers
  - Circuit: non-scenario, **non-SIMD**
  - Goal: Have a non-SIMD circuit with inputs that can be XORed


## Face recognition models:

WIP
