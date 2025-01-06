cmake_minimum_required(VERSION 3.13)
project(flatter LANGUAGES CXX)

# Flags
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wno-error=overloaded-virtual)
add_compile_options(-Wno-error=unused-but-set-variable)


# includes
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include_directories(./include)

#for homebrew's libs
if(APPLE)
    include_directories(/opt/homebrew/include)
    include_directories(/opt/homebrew/opt/libomp/include)
    link_directories(/opt/homebrew/lib)
    link_directories(/opt/homebrew/opt/libomp/lib)
endif()

add_subdirectory(src)
add_subdirectory(apps)

