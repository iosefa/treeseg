cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0144 NEW)

project(treeseg)

# Find packages
find_package(PCL 1.10 REQUIRED)
find_package(Armadillo 9.8 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem iostreams serialization program_options)
# Include directories
include_directories(
		${PCL_INCLUDE_DIRS}
		${ARMADILLO_INCLUDE_DIRS}
		${Boost_INCLUDE_DIRS}
		${CMAKE_SOURCE_DIR}/include
		${CMAKE_SOURCE_DIR}/include/riegl
)

# Link directories
link_directories(
		${CMAKE_SOURCE_DIR}/lib
		${CMAKE_SOURCE_DIR}/lib/riegl
		${Boost_LIBRARY_DIRS}
)

# Define treeseg library
add_library(treeseg SHARED ${CMAKE_SOURCE_DIR}/src/treeseg.cpp)
target_link_libraries(treeseg ${PCL_LIBRARIES} ${Boost_LIBRARIES})

# Define leafsep library
add_library(leafsep SHARED ${CMAKE_SOURCE_DIR}/src/leafsep.cpp)
target_link_libraries(leafsep treeseg ${PCL_LIBRARIES} ${ARMADILLO_LIBRARIES})

# Add executables and link libraries
add_executable(plotcoords ${CMAKE_SOURCE_DIR}/src/plotcoords.cpp)
target_link_libraries(plotcoords ${PCL_LIBRARIES})

add_executable(nearestneighbour ${CMAKE_SOURCE_DIR}/src/nearestneighbour.cpp)
target_link_libraries(nearestneighbour treeseg ${PCL_LIBRARIES})

add_executable(downsample ${CMAKE_SOURCE_DIR}/src/downsample.cpp)
target_link_libraries(downsample treeseg ${PCL_LIBRARIES})

add_executable(thin ${CMAKE_SOURCE_DIR}/src/thin.cpp)
target_link_libraries(thin treeseg ${PCL_LIBRARIES})

add_executable(getdtmslice ${CMAKE_SOURCE_DIR}/src/getdtmslice.cpp)
target_link_libraries(getdtmslice treeseg ${PCL_LIBRARIES})

add_executable(findstems ${CMAKE_SOURCE_DIR}/src/findstems.cpp)
target_link_libraries(findstems treeseg ${PCL_LIBRARIES})

add_executable(segmentstem ${CMAKE_SOURCE_DIR}/src/segmentstem.cpp)
target_link_libraries(segmentstem treeseg ${PCL_LIBRARIES})

add_executable(getcrownvolume ${CMAKE_SOURCE_DIR}/src/getcrownvolume.cpp)
target_link_libraries(getcrownvolume treeseg ${PCL_LIBRARIES})

add_executable(segmentcrown ${CMAKE_SOURCE_DIR}/src/segmentcrown.cpp)
target_link_libraries(segmentcrown treeseg ${PCL_LIBRARIES})

add_executable(sepwoodleaf ${CMAKE_SOURCE_DIR}/src/sepwoodleaf.cpp)
target_link_libraries(sepwoodleaf treeseg leafsep ${PCL_LIBRARIES})

add_executable(pcdPointTreeseg2txt ${CMAKE_SOURCE_DIR}/src/pcdPointTreeseg2txt.cpp)
target_link_libraries(pcdPointTreeseg2txt treeseg ${PCL_LIBRARIES})

add_executable(txtPointTreeseg2pcd ${CMAKE_SOURCE_DIR}/src/txtPointTreeseg2pcd.cpp)
target_link_libraries(txtPointTreeseg2pcd treeseg ${PCL_LIBRARIES})

add_executable(pcdPointXYZRGB2txt ${CMAKE_SOURCE_DIR}/src/pcdPointXYZRGB2txt.cpp)
target_link_libraries(pcdPointXYZRGB2txt ${PCL_LIBRARIES})

# Special handling for rxp2pcd executable on Linux
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	if(EXISTS "${CMAKE_SOURCE_DIR}/lib/riegl/libscanifc-mt.so")
		add_executable(rxp2pcd ${CMAKE_SOURCE_DIR}/src/rxp2pcd.cpp)
		target_link_libraries(rxp2pcd treeseg scanifc-mt)
	else()
		message(WARNING "rxp executables not building because either RIEGL libs/headers not found, or not a compatible OS.")
	endif()
endif()
