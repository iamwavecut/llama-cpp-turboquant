set(TARGET llama-spectral-calibrate)
add_executable(${TARGET} spectral-calibrate.cpp)
target_link_libraries(${TARGET} PRIVATE common)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
