# Build integration for the Project 0 C++ programs.
#
# Put this file plus two_seq.cpp and driver.cpp in:
#     llama.cpp/examples/project0/
# Then add this line to  llama.cpp/examples/CMakeLists.txt :
#     add_subdirectory(project0)
# Rebuild from the repo root:
#     cmake --build build --config Release -j
# Binaries appear at:
#     build/bin/Release/llama-two-seq.exe     (Windows)   or  build/bin/llama-two-seq
#     build/bin/Release/llama-doc-driver.exe              or  build/bin/llama-doc-driver

add_executable(llama-two-seq two_seq.cpp)
target_link_libraries(llama-two-seq PRIVATE llama)
target_compile_features(llama-two-seq PRIVATE cxx_std_17)

add_executable(llama-doc-driver driver.cpp)
target_link_libraries(llama-doc-driver PRIVATE llama)
target_compile_features(llama-doc-driver PRIVATE cxx_std_17)
