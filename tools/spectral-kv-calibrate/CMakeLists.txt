set(TARGET llama-spectral-kv-calibrate)
add_executable(${TARGET} spectral-kv-calibrate.cpp)
target_link_libraries(${TARGET} PRIVATE common)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
