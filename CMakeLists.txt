cmake_minimum_required(VERSION 3.5)
project(ml_classifiers)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})

rosidl_generate_interfaces(pendulum_msgs
  "msg/ClassDataPoint.msg"
  "srv/AddClassData.srv"
  "srv/ClassifyData.srv"
  "srv/ClearClassifier.srv"
  "srv/CreateClassifier.srv"
  "srv/LoadClassifier.srv"
  "srv/SaveClassifier.srv"
  "srv/TrainClassifier.srv"
  DEPENDENCIES builtin_interfaces std_msgs
)

add_definitions(${EIGEN3_DEFINITIONS})

include_directories(
  include
  src
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(ZeroClassifier SHARED
  src/zero_classifier.cpp) 
add_library(NearestNeighborClassifier SHARED
  src/nearest_neighbor_classifier.cpp)
add_library(SVMClassifier SHARED
  src/svm_classifier.cpp
  src/libsvm-3.14/svm.cpp)

add_executable(classifier_server
  nodes/classifier_server.cpp)

ament_target_dependencies(classifier_server
  "rclcpp")

rosidl_target_interfaces(classifier_server
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp")

install(
  TARGETS
    SVMClassifier
    NearestNeighborClassifier
    ZeroClassifier
    classifier_server
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_dependencies(ament_cmake)
ament_export_dependencies(rclcpp)
ament_export_dependencies(class_loader)

ament_export_include_directories(include)

ament_export_libraries(SVMClassifier)
ament_export_libraries(NearestNeighborClassifier)
ament_export_libraries(ZeroClassifier)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
