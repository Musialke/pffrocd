
add_executable(cos_sim_simd_test cos_sim_simd.cpp)
target_link_libraries(cos_sim_simd_test ABY::aby ENCRYPTO_utils::encrypto_utils)
