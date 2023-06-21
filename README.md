# pffrocd
Privacy-Friendly Face Recognition On Constrained Devices

## ABY

1. Enter the framework directory: `cd ABY/`
2. Create and enter the build directory: `mkdir build && cd build`
3. Use CMake to configure the build with example applications: ```cmake .. -DABY_BUILD_EXE=On```
4. Call `make` in the build directory. You can find the build executables and libraries in the directories `bin/` and `lib/`, respectively.

The project-specific examples are in `src/kamil/`

Working examples:
- `float_add`: add two random floats

To run the examples (both SERVER and CLIENT) in one terminal:

```bash
./cos_sim_test -r 1 & (./cos_sim_test -r 0 2>&1 > /dev/null)
```

## Face recognition models:

WIP
