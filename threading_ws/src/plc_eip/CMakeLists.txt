cmake_minimum_required(VERSION 3.8)
project(plc_eip)


set(CMAKE_BUILD_TYPE RELEASE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cherry_interfaces REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.

# add the Cipster library

include(ExternalProject)





include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/plc_eip)
# add_subdirectory(
# ${CMAKE_CURRENT_SOURCE_DIR}/include/CIPster/source)




set( CIPSTER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/CIPster)
# These variables would typically be found by a CMake find EIP module, but set
# them manually here

    

set( USER_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/plc_eip
    CACHE PATH "Location of user specific include file (cipster_user_conf.h)"
    )

set( EIP_INCLUDE_DIR ${CIPSTER_DIR}/source/src )
# set( EIP_LIBRARIES   ${PREFIX}/libeip.a )
# set (EIP_LIBRARIES   libeip.a)


    # add_definitions( -std=c++0x )

    # PREFIX is for ExternalProject_Add, and tells where to build CIPster as a sub project:
    # below our current out of tree build directory.
    set( PREFIX ${CMAKE_CURRENT_BINARY_DIR}/build-CIPster )
    
    # build CIPster as a nested project, the result of which is libeip.a
    # in directory ${PREFIX}
    ExternalProject_Add( eip
        PREFIX ${PREFIX}
        SOURCE_DIR ${CIPSTER_DIR}/source
        CONFIGURE_COMMAND
            ${CMAKE_COMMAND}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${PREFIX}
            -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
            -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DUSER_INCLUDE_DIR=${USER_INCLUDE_DIR}
            ${TRACE_SPEC}       # empty for non Debug CMAKE_BUILD_TYPE
            <SOURCE_DIR>
        BUILD_COMMAND make
    
        INSTALL_COMMAND make install
        )
    
    #message( "TRACE_SPEC=${TRACE_SPEC}" )

    set( EIP_LIBRARIES   ${PREFIX}/libeip.a )

    message(plc_eip EIP_LIBRARIES="${EIP_LIBRARIES}")
    message(plc_eip USER_INCLUDE_DIR="${USER_INCLUDE_DIR}")
    message(plc_eip CIPSTER_DIR="${CIPSTER_DIR}")
    
set( PGM_SRCS
${PROJECT_SOURCE_DIR}/src/plc_eip.cpp
    )

    if( CMAKE_BUILD_TYPE STREQUAL Debug )
    add_definitions( -DCIPSTER_WITH_TRACES -DCIPSTER_TRACE_LEVEL=15 )
    set( TRACE_SPEC "-DCIPster_TRACES=ON" )
endif()

    if( NOT APPLE )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY}hidden" )
    endif()
    if( NOT APPLE )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}" )
    endif()
    
    
    include_directories(
        ${EIP_INCLUDE_DIR}
        ${USER_INCLUDE_DIR}
        )

# target_link_libraries(plc_eip ${EIP_INCLUDE_DIR}/cipster_api.h)

# add_executable(plc_eip ${PROJECT_SOURCE_DIR}/src/publisher_member_function.cpp ${PROJECT_SOURCE_DIR}/src/sampleapplication.cc)
add_executable(plc_eip ${PROJECT_SOURCE_DIR}/src/plc_eip.cpp )

ament_target_dependencies(plc_eip rclcpp std_msgs cherry_interfaces)

target_include_directories(plc_eip PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIP_INCLUDE_DIR}
  ${EIP_LIBRARIES})

target_link_libraries(plc_eip ${EIP_LIBRARIES})

install(TARGETS
  plc_eip
  DESTINATION lib/${PROJECT_NAME})

  # add_dependencies( plc_eip ${EIP_LIBRARIES} )
  add_dependencies( plc_eip eip )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
