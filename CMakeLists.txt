cmake_minimum_required(VERSION 3.5)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -Wall")

find_package(PNG REQUIRED)
find_package(png++ REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
find_package(Threads REQUIRED)  # wtf gcc
include_directories(
    ${PNG_INCLUDE_DIR},
    ${png++_INCLUDE_DIR})

set(targets "bas;bram;sleeve;refracting")
foreach(trg ${targets})
  add_executable(${trg} ${trg}.cpp)
  target_link_libraries(${trg}
      ${png++_LIBRARIES}
      ${PNG_LIBRARY}
      ${Boost_LIBRARIES}
      ${CMAKE_THREAD_LIBS_INIT})
  set_property(TARGET ${trg} PROPERTY CXX_STANDARD 14)
endforeach()
