cmake_minimum_required(VERSION 3.8)
project(topsbot_mobilenet_cls)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(topsbot_ai_infer REQUIRED)
find_package(topsbot_cv REQUIRED)
find_package(OpenCV REQUIRED)

if(CMAKE_SYSROOT)
  set(TOPSBOT_MOBILENET_SYSTEM_ROOT "${CMAKE_SYSROOT}/usr/local")
else()
  set(TOPSBOT_MOBILENET_SYSTEM_ROOT "/usr/local")
endif()
set(TOPSBOT_MOBILENET_SYSTEM_INCLUDE "${TOPSBOT_MOBILENET_SYSTEM_ROOT}/include")
set(TOPSBOT_MOBILENET_SYSTEM_LIB "${TOPSBOT_MOBILENET_SYSTEM_ROOT}/lib")

add_library(topsbot_mobilenet_utils SHARED
  src/mobilenet_utils.cpp
)
target_include_directories(topsbot_mobilenet_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${TOPSBOT_MOBILENET_SYSTEM_INCLUDE}
)
ament_target_dependencies(topsbot_mobilenet_utils topsbot_ai_infer)

add_library(topsbot_mobilenet_dma SHARED
  src/dma_buffer.cpp
  src/dma_preprocess.cpp
)
target_include_directories(topsbot_mobilenet_dma PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TOPSBOT_MOBILENET_SYSTEM_INCLUDE}
)
target_link_directories(topsbot_mobilenet_dma PUBLIC
  ${TOPSBOT_MOBILENET_SYSTEM_LIB}
)
target_link_options(topsbot_mobilenet_dma PUBLIC
  "-Wl,-rpath-link,${TOPSBOT_MOBILENET_SYSTEM_LIB}"
)
target_link_libraries(topsbot_mobilenet_dma
  tacocv
  taco_jpegdec
  taco_ppvdec
  enc24
  dmabufheap
  tacosys
)
ament_target_dependencies(topsbot_mobilenet_dma topsbot_cv)

add_library(topsbot_mobilenet_backend SHARED
  src/mobilenet_backend.cpp
)
target_include_directories(topsbot_mobilenet_backend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${TOPSBOT_MOBILENET_SYSTEM_INCLUDE}
)
target_link_directories(topsbot_mobilenet_backend PUBLIC
  ${TOPSBOT_MOBILENET_SYSTEM_LIB}
)
target_link_libraries(topsbot_mobilenet_backend
  topsbot_mobilenet_utils
  topsbot_mobilenet_dma
  ta_runtime
)
ament_target_dependencies(topsbot_mobilenet_backend
  pluginlib
  topsbot_ai_infer
  topsbot_cv
)

add_executable(mobilenet_file_cli
  src/mobilenet_file_cli.cpp
)
target_include_directories(mobilenet_file_cli PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mobilenet_file_cli ${OpenCV_LIBS})
ament_target_dependencies(mobilenet_file_cli
  ament_index_cpp
  topsbot_ai_infer
)

add_executable(mobilenet_eval_cli
  test/mobilenet_eval_cli.cpp
)
ament_target_dependencies(mobilenet_eval_cli
  ament_index_cpp
  topsbot_ai_infer
)

add_executable(mobilenet_tacv_compare
  test/mobilenet_tacv_compare.cpp
)
target_include_directories(mobilenet_tacv_compare PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${TOPSBOT_MOBILENET_SYSTEM_INCLUDE}
)
target_link_libraries(mobilenet_tacv_compare
  topsbot_mobilenet_dma
  ${OpenCV_LIBS}
)
ament_target_dependencies(mobilenet_tacv_compare topsbot_cv)

install(TARGETS
  topsbot_mobilenet_utils
  topsbot_mobilenet_dma
  topsbot_mobilenet_backend
  mobilenet_file_cli
  mobilenet_eval_cli
  mobilenet_tacv_compare
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/download_models.sh
  scripts/ensure_models.sh
  scripts/model_manager.py
  DESTINATION share/${PROJECT_NAME}/scripts
)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

pluginlib_export_plugin_description_file(topsbot_ai_infer mobilenet_plugins.xml)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_mobilenet_utils test/test_mobilenet_utils.cpp)
  if(TARGET test_mobilenet_utils)
    target_include_directories(test_mobilenet_utils PRIVATE
      ${TOPSBOT_MOBILENET_SYSTEM_INCLUDE}
    )
    target_link_libraries(test_mobilenet_utils topsbot_mobilenet_utils)
    ament_target_dependencies(test_mobilenet_utils topsbot_ai_infer)
  endif()

  ament_add_gtest(test_dma_buffer test/test_dma_buffer.cpp)
  if(TARGET test_dma_buffer)
    target_include_directories(test_dma_buffer PRIVATE
      ${TOPSBOT_MOBILENET_SYSTEM_INCLUDE}
    )
    target_link_libraries(test_dma_buffer topsbot_mobilenet_dma)
    ament_target_dependencies(test_dma_buffer topsbot_cv)
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(
  topsbot_mobilenet_utils
  topsbot_mobilenet_dma
  topsbot_mobilenet_backend)
ament_export_dependencies(pluginlib topsbot_ai_infer topsbot_cv)
ament_package()
