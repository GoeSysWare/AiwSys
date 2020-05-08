file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

add_library(${UTIL_TARGET} 
	${HEADER_FILES}
	${SOURCE_FILES}
)

target_link_libraries (${UTIL_TARGET}  
    Qt5::Widgets Qt5::Gui Qt5::Core Qt5::OpenGL
	${Boost_LIBRARIES}
	${PROTOBUF_LIBRARIES}
    ${OpenCV_LIBRARIES}
	asound
)

set_target_properties(${UTIL_TARGET} PROPERTIES FOLDER "my_library")