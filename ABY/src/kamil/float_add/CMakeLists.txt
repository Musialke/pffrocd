
add_executable(float_add_test float_add_test.cpp common/float_add.cpp)
target_link_libraries(float_add_test ABY::aby ENCRYPTO_utils::encrypto_utils)
