cmake_minimum_required(VERSION 3.16)
project(orbiting)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(orbiting orbital_dynamics_simulator.cpp)

# Optional: enable some optimization and warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
    target_compile_options(orbiting PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()
