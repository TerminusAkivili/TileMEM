cmake_minimum_required(VERSION 3.26)

project(TileMEM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(TILEMEM_SM "120" CACHE STRING "Target NVIDIA SM architecture")

enable_testing()

find_package(CUDAToolkit REQUIRED)

add_library(tilemem_core
  src/config.cc
  src/io_engine.cc
  src/manifest.cc
  src/memory_manager.cc
  src/metrics.cc
  src/scheduler.cc
  src/queues.cc
  src/trace.cc
  src/types.cc
  src/fp4_gemm.cc
  kernels/fp4_gemm_sm120.cu
  kernels/tilepo_moe_hot_tile_bf16.cu
  kernels/tilepo_moe_hot_tile_fp8.cu
  kernels/tilepo_moe_hot_tile_mxfp4.cu
)
target_include_directories(tilemem_core PUBLIC include)
target_link_libraries(tilemem_core PUBLIC CUDA::cudart)
set_target_properties(tilemem_core PROPERTIES CUDA_ARCHITECTURES ${TILEMEM_SM})

add_executable(tilemem_generate_demo_data tools/generate_demo_data/main.cc)
target_link_libraries(tilemem_generate_demo_data PRIVATE tilemem_core)

add_executable(tilemem_replay tools/replay/main.cc)
target_link_libraries(tilemem_replay PRIVATE tilemem_core)

add_executable(tilemem_tests tests/test_tilemem.cpp)
target_link_libraries(tilemem_tests PRIVATE tilemem_core)
set_target_properties(tilemem_tests PROPERTIES CUDA_ARCHITECTURES ${TILEMEM_SM})

add_test(NAME tilemem_tests COMMAND tilemem_tests)
add_test(NAME tilemem_generate_demo_data
  COMMAND tilemem_generate_demo_data --config ${CMAKE_SOURCE_DIR}/configs/demo.toml --out ${CMAKE_BINARY_DIR}/demo_data)
add_test(NAME tilemem_replay_tilemem
  COMMAND tilemem_replay --config ${CMAKE_SOURCE_DIR}/configs/demo.toml --data-root ${CMAKE_BINARY_DIR}/demo_data --policy tilemem)
add_test(NAME tilemem_replay_on_demand
  COMMAND tilemem_replay --config ${CMAKE_SOURCE_DIR}/configs/demo.toml --data-root ${CMAKE_BINARY_DIR}/demo_data --policy on_demand)
add_test(NAME tilemem_replay_lru
  COMMAND tilemem_replay --config ${CMAKE_SOURCE_DIR}/configs/demo.toml --data-root ${CMAKE_BINARY_DIR}/demo_data --policy lru)
add_test(NAME tilemem_replay_layerwise
  COMMAND tilemem_replay --config ${CMAKE_SOURCE_DIR}/configs/demo.toml --data-root ${CMAKE_BINARY_DIR}/demo_data --policy layerwise_prefetch)
set_tests_properties(tilemem_replay_tilemem tilemem_replay_on_demand
  tilemem_replay_lru tilemem_replay_layerwise
  PROPERTIES DEPENDS tilemem_generate_demo_data)

add_test(NAME tilepo_ablation COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/tests/assert_tilepo_ablation.py)
add_test(NAME tilemem_python_sdk COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/tests/assert_tilemem_sdk.py)
add_test(NAME tilemem_cli COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/tests/assert_tilemem_cli.py)
add_test(NAME tilemem_agent_skills COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/tests/assert_agent_skills.py)
add_test(NAME tilemem_checkpoint_integration COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/tests/assert_checkpoint_integration.py)
add_test(NAME tilemem_industrial_quickstart COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/tests/assert_tilemem_industrial_quickstart.py)
add_test(NAME tilemem_removed_predictor_traces COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/tests/assert_removed_predictor_traces.py)
set_tests_properties(
  tilepo_ablation
  tilemem_python_sdk
  tilemem_cli
  tilemem_agent_skills
  tilemem_checkpoint_integration
  tilemem_industrial_quickstart
  tilemem_removed_predictor_traces
  PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
