cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)
project(cppai-test)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/f5e592d8ee5ffb1d9af5be7f715ce3576b8bf9c4.zip)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
find_package(xtensor REQUIRED)

enable_testing()

set(TESTS
    hello_test
    knn_test
)

foreach(t ${TESTS})
    add_executable(${t} src/${t}.cpp)
    target_link_libraries(${t} PUBLIC gtest_main xtensor)
    target_include_directories(${t} PUBLIC ${xtensor_INCLUDE_DIRS})
    gtest_discover_tests(${t})
endforeach(t)

