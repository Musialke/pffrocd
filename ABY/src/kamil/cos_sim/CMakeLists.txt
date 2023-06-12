
add_executable(cos_sim_test cos_sim_test.cpp common/cos_sim.cpp)
target_link_libraries(cos_sim_test ABY::aby ENCRYPTO_utils::encrypto_utils)
