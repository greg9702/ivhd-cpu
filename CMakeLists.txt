CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(ivhd-cpu)

SET(CMAKE_CXX_STANDARD 17)

FIND_PACKAGE(Boost REQUIRED)

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/dependencies/
    ${Boost_INCLUDE_DIR}
)

FILE(GLOB SOURCES "src/*.cpp" "src/caster/*.cpp")
ADD_EXECUTABLE( ivhd-cpu 
    ${SOURCES}
        src/caster/caster_sgd.h src/caster/caster_sgd.cpp src/caster/caster_momentum.cpp src/caster/caster_momentum.h src/caster/caster_adam.cpp src/caster/caster_adam.h src/caster/constants.h)

TARGET_LINK_LIBRARIES( ivhd-cpu
    ${Boost_LIBRARIES}
)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")