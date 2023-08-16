# pffrocd
Privacy-Friendly Face Recognition On Constrained Devices

## Complete guide to set up a host for tests:

Required:

`python3 python3.10-venv g++ make cmake libgmp-dev libssl-dev libboost-all-dev ffmpeg libsm6 libxext6`

1. Clone the repo after adding the machine's ssh key as a deploy key

```sh
git clone git@github.com:Musialke/pffrocd.git
cd pffrocd 
```
2. Create a virtualenv in the folder (assuming Python3 with venv is installed) and activate it

```sh
python3 -m venv env
. env/bin/activate
```
3. Install required packages

```sh
pip install -vr requirements.txt
```

## SFE

Setting up:

1. Enter the framework directory: `cd ABY/`
2. Create and enter the build directory: `mkdir build && cd build`
3. Use CMake to configure the build with example applications: ```cmake .. -DABY_BUILD_EXE=On```
4. Call `make` in the build directory. You can find the build executables and libraries in the directories `bin/` and `lib/`, respectively.

The project-specific examples are in `src/kamil/`

To run the examples run the corresponding python script located in `pyscripts/`

1. Unpack the split database with face images by running
```sh
cat lfw.tgz.parta* | tar -xzv
```
2. Make sure you have Python3 on your system with venv installed
3. Create a new virtual environment, activate it and install required packages
```sh
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```
4. Run a script, for example
```sh
python3 pyscripts/cos_dist_float_nscen_simd.py
```

### Possible errors and solutions:

`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`
fix:
```sh
sudo apt update && sudo apt install ffmpeg libsm6 libxext6  -y
```

`v2.error: OpenCV(4.7.0) /io/opencv/modules/dnn/src/onnx/onnx_importer.cpp:275: error: (-210:Unsupported format or combination of formats) Failed to parse ONNX model: /home/dietpi/.deepface/weights/face_recognition_sface_2021dec.onnx in function 'ONNXImporter'` 
Link to weights for SFace is missing. fix:
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
