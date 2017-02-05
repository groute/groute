cmake_minimum_required (VERSION 2.8)
project (groute)

############## DEPENDENCIES ##############
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDA 7 REQUIRED)

# GFlags
add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)

# GTest
find_package(GTest QUIET)
if(NOT GTEST_FOUND)
  add_subdirectory(deps/googletest/googletest)
endif()

# Try to find METIS
find_package(METIS QUIET)
if (NOT METIS_FOUND)
  # Otherwise, try to find a local copy
  if (EXISTS "${PROJECT_SOURCE_DIR}/metis/")
    set(GKLIB_PATH ${PROJECT_SOURCE_DIR}/metis/GKlib CACHE PATH "path to GKlib" FORCE)
    add_subdirectory(metis)
    set(METIS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/metis/include)
    set(METIS_LIBRARIES metis)
    set(METIS_FOUND "true")
  else()
    message(WARNING "Compiling without METIS partitioning. To install METIS, run 'sudo apt-get install libmetis-dev' or manually download metis and extract to a subdirectory called 'metis'")

    set(METIS_INCLUDE_DIRS "")
    set(METIS_LIBRARIES "")
    
  endif()
endif()

if (METIS_FOUND)
  message("-- Found METIS: ${METIS_LIBRARIES}")
  add_definitions(-DHAVE_METIS)
endif()


# CUB
include_directories(${PROJECT_SOURCE_DIR}/deps/cub)

############## BUILD ##############

include_directories(${PROJECT_SOURCE_DIR}/include)

set(EXTRA_LIBS gflags_static pthread ${METIS_LIBRARIES} ${CUDA_TOOLKIT_TARGET_DIR}/lib64/stubs/libcuda.so ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libnvToolsExt.so)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-g;-lineinfo;-Xcompiler;-ggdb;-std=c++11)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG;-std=c++11)
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DNDEBUG")

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${METIS_INCLUDE_DIRS})
cuda_add_executable(cc src/utils/parser.cpp src/utils/utils.cpp
                    samples/cc/cc_async.cu samples/cc/main.cpp)
target_link_libraries(cc ${EXTRA_LIBS})

cuda_add_executable(bfs src/utils/parser.cpp src/utils/utils.cpp src/groute/graphs/csr_graph.cpp 
                    samples/bfs/bfs_async.cu samples/bfs/bfs_async_opt.cu
                    samples/bfs/bfs_host.cpp 
					samples/bfs/main.cpp)
target_link_libraries(bfs ${EXTRA_LIBS})

cuda_add_executable(sssp src/utils/parser.cpp src/utils/utils.cpp src/groute/graphs/csr_graph.cpp
                    samples/sssp/sssp_async.cu samples/sssp/sssp_async_opt.cu
                    samples/sssp/sssp_host.cpp 
					samples/sssp/main.cpp)
target_link_libraries(sssp ${EXTRA_LIBS})

cuda_add_executable(pr src/utils/parser.cpp src/utils/utils.cpp src/groute/graphs/csr_graph.cpp
					samples/pr/pr_async.cu samples/pr/pr_async_opt.cu samples/pr/pr_host.cpp 
					samples/pr/main.cpp)
target_link_libraries(pr ${EXTRA_LIBS})

cuda_add_executable(pbf samples/pbf/pbf_async.cu samples/pbf/main.cpp)
target_link_libraries(pbf ${EXTRA_LIBS})


# Unit tests
enable_testing()

# GTest directory settings
if(NOT GTEST_FOUND)
  include_directories(
    ${gtest_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR})
  link_directories(${gtest_BINARY_DIR}/src)
endif()
				
# Groute async component tests
cuda_add_executable(async-tests test/async-tests/main.cpp
                    test/async-tests/router_test.cu
                    test/async-tests/worklist_test.cu
                    test/async-tests/circular_worklist_test.cu)

target_link_libraries(async-tests gtest pthread dl ${EXTRA_LIBS})
cuda_add_cublas_to_target(async-tests)
				
# Groute async component micro-benchmark tests
cuda_add_executable(mb-tests test/micro-benchmarks/main.cpp
                    test/micro-benchmarks/timed_kernel.cu test/micro-benchmarks/high_priority_copy.cu)

target_link_libraries(mb-tests gtest pthread)
cuda_add_cublas_to_target(mb-tests)
