cmake_minimum_required(VERSION 3.10)

#header files directory
include_directories(${ljmd_SOURCE_DIR}/include)

if(ENABLE_MPI)
	include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif()

#target library name and sources
add_library(ljmd_phys_lib integration.c force.c ekin.c utils.c)
target_link_libraries(ljmd_phys_lib PRIVATE m)

#list of compilation options
list(APPEND phys_flags -Wall -g -O3 -ffast-math -fomit-frame-pointer)

#compile options
target_compile_options (ljmd_phys_lib PRIVATE ${phys_flags}  )


if(ENABLE_MPI)
	target_link_libraries(ljmd_phys_lib PRIVATE 
					${MPI_C_LIBRARIES}
					${MPI_C_LINK_FLAGS} 
					ljmd_mpi_util_lib
					)
	target_compile_definitions(ljmd_phys_lib PRIVATE MPI_ENABLED=1 )
elseif(ENABLE_OMP)
	# target library name and sources (for force.c)
	target_link_libraries(ljmd_phys_lib PRIVATE m OpenMP::OpenMP_C)

	target_compile_definitions(ljmd_phys_lib PRIVATE _OMP_NAIVE  _OMP_3RD_LAW)
	
endif()



get_target_property(PHYS_COMPILE_OPT ljmd_phys_lib COMPILE_OPTIONS)
message("-- ljmd_phys_lib compiler flags are: ${PHYS_COMPILE_OPT}")
