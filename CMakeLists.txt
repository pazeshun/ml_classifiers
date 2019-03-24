project(ml_classifiers)

set(ROS_VERSION $ENV{ROS_VERSION})

# Version-independent instructions
find_package(Eigen3 REQUIRED)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
add_definitions(${EIGEN3_DEFINITIONS})

if(${ROS_VERSION} EQUAL 1)

  cmake_minimum_required(VERSION 2.8.3)

  # Default to C++11
  if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
  endif()

  find_package(catkin REQUIRED
    COMPONENTS
    message_generation
    std_msgs
    roscpp
    pluginlib
    roslint
  )

  include_directories(
    include
    src
    ${EIGEN3_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
  )

  add_message_files(FILES
    ClassDataPoint.msg
    DIRECTORY msg
  )

  add_service_files(FILES
    AddClassData.srv
    ClassifyData.srv
    ClearClassifier.srv
    CreateClassifier.srv
    LoadClassifier.srv
    SaveClassifier.srv
    TrainClassifier.srv
    DIRECTORY srv
  )

  generate_messages(DEPENDENCIES std_msgs)

  catkin_package(
    DEPENDS Eigen3
    CATKIN_DEPENDS message_runtime std_msgs roscpp pluginlib 
    INCLUDE_DIRS include
    LIBRARIES ZeroClassifier NearestNeighborClassifier SVMClassifier
  )

elseif(${ROS_VERSION} EQUAL 2)

  cmake_minimum_required(VERSION 3.5)

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

  include_directories(
    include
    src
    ${EIGEN3_INCLUDE_DIRS}
  )

  rosidl_generate_interfaces(${PROJECT_NAME}_msgs
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

endif()

add_library(ZeroClassifier SHARED
  src/zero_classifier.cpp) 
add_library(NearestNeighborClassifier SHARED
  src/nearest_neighbor_classifier.cpp)
add_library(SVMClassifier SHARED
  src/svm_classifier.cpp
  src/libsvm-3.14/svm.cpp)

if(${ROS_VERSION} EQUAL 1)

  add_executable(classifier_server
    nodes/classifier_server.cpp)

  target_link_libraries(classifier_server ${catkin_LIBRARIES})
  add_dependencies(classifier_server ${PROJECT_NAME}_generate_messages_cpp)

  roslint_cpp(
    include/ml_classifiers/classifier.h
    include/ml_classifiers/nearest_neighbor_classifier.h
    include/ml_classifiers/svm_classifier.h
    include/ml_classifiers/zero_classifier.h
    nodes/classifier_server.cpp
    src/nearest_neighbor_classifier.cpp
    src/svm_classifier.cpp
    src/zero_classifier.cpp
  )

  install(TARGETS classifier_server ZeroClassifier NearestNeighborClassifier SVMClassifier
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  if(CATKIN_ENABLE_TESTING)
    roslint_add_test()
  endif()

elseif(${ROS_VERSION} EQUAL 2)

  ament_target_dependencies(classifier_server
    "rclcpp")

  rosidl_target_interfaces(classifier_server
    ${PROJECT_NAME}_msgs
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

  pluginlib_export_plugin_description_file(ml_classifiers
    "default_classifiers_plugin.xml")

  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)

    ament_lint_auto_find_test_dependencies()
  endif()

  ament_package()

endif()
