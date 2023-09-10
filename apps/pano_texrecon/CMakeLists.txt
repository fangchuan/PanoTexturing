# opencv
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# MESSAGE(STATUS "OpenCV version is " ${OpenCV_VERSION})

#find Open3D
find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})

# find glog
# find_package(glog REQUIRED)
# include_directories(${GLOG_INCLUDE_DIRS})

# file (GLOB HEADERS "*.h")
# file (GLOB SOURCES "[^_]*.cpp")

set(BIN panorecons)

add_executable(${BIN} test.cpp
                     pano_recons.cpp 
                     cubemap.cpp
                     panorama.cpp)
# add_executable(${BIN} test.cpp)
# set_property(TARGET ${BIN} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
add_dependencies(${BIN} ext_mve)
target_link_libraries(${BIN} tex ${TBB_LIBRARIES}  -lmve -lmve_util ${Open3D_LIBRARIES})
install(TARGETS ${BIN} RUNTIME DESTINATION bin)