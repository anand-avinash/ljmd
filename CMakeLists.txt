cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(ljmd VERSION 1.1 LANGUAGES C)



if(ENABLE_MPI)
			find_package(MPI REQUIRED) 
			set( tgt_name mpi)
			set( tgt_exe_name "ljmd-mpi.x")
else()			
			set( tgt_name serial)
			set( tgt_exe_name "ljmd-serial.x")
endif()

#specify target exe name and source
add_executable(${tgt_name}  src/ljmd.c)
set_target_properties(${tgt_name} PROPERTIES OUTPUT_NAME ${tgt_exe_name})


#subdirectories to run their own cmake files 
list(APPEND sub_dirs 
					${ljmd_SOURCE_DIR}/src/common  
					${ljmd_SOURCE_DIR}/src/io  
					${ljmd_SOURCE_DIR}/src/physics 
					)

if(ENABLE_TESTING)
					enable_testing()
					enable_language(CXX)
					list(APPEND sub_dirs ${ljmd_SOURCE_DIR}/tests )
endif()


if(ENABLE_MPI)
					list(APPEND sub_dirs 
						${ljmd_SOURCE_DIR}/src/mpi/mpi_utils
						${ljmd_SOURCE_DIR}/src/mpi/mpi_comm
					)
endif()


foreach(subdir ${sub_dirs})
  					add_subdirectory(${subdir})
endforeach()


#include dir for the header files
include_directories(${ljmd_SOURCE_DIR}/include)
if(ENABLE_MPI)
	include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif()


#link libraries and the lib source files
target_link_libraries(${tgt_name}  PRIVATE 
					ljmd_phys_lib
					ljmd_io_lib
					ljmd_common_lib
					)


if(ENABLE_MPI)
	target_link_libraries(${tgt_name} PRIVATE 
					${MPI_C_LIBRARIES}
					${MPI_C_LINK_FLAGS} 
					ljmd_mpi_util_lib
					ljmd_mpi_comm_lib
					)

endif()



#compilation options list
list(APPEND main_flags -Wall -g -O3 -ffast-math -msse3 -fexpensive-optimizations )

#compile options
target_compile_options (${tgt_name}  PRIVATE ${main_flags}  )
get_target_property(LJMD_COMPILE_OPT ${tgt_name}  COMPILE_OPTIONS)
message("-- ljmd main compiler flags are: ${LJMD_COMPILE_OPT}")


target_compile_definitions(${tgt_name}  PRIVATE LJMD_VERSION=1.0)
if(ENABLE_MPI)
	target_compile_definitions(${tgt_name}  PRIVATE MPI_ENABLED=1)
endif()
if(ENABLE_TIMING)
	target_compile_definitions(${tgt_name}  PRIVATE TIMING=1)
endif()




##########################
# _OMP_NAIVE
##########################

#specify target exe name and source
add_executable(omp_naive src/ljmd.c)
set_target_properties(omp_naive PROPERTIES OUTPUT_NAME "ljmd_omp_naive.x")

#link libraries and the lib source files
find_package(OpenMP REQUIRED)
target_link_libraries(omp_naive PRIVATE ljmd_phys_lib_omp_naive ljmd_force_lib_omp_naive	ljmd_io_lib ljmd_common_lib OpenMP::OpenMP_C)

#compile options
target_compile_options (omp_naive PRIVATE ${main_flags})
target_compile_definitions(omp_naive PRIVATE LJMD_VERSION=1.0 _OMP_NAIVE)

##########################
# _OMP_3RD_LAW
##########################

#specify target exe name and source
add_executable(omp_3rd_law src/ljmd.c)
set_target_properties(omp_3rd_law PROPERTIES OUTPUT_NAME "ljmd_omp_3rd_law.x")

#link libraries and the lib source files
find_package(OpenMP REQUIRED)
target_link_libraries(omp_3rd_law PRIVATE ljmd_phys_lib_omp_naive ljmd_force_lib_omp_3rd_law	ljmd_io_lib ljmd_common_lib OpenMP::OpenMP_C)

#compile options
target_compile_options (omp_3rd_law PRIVATE ${main_flags})
target_compile_definitions(omp_3rd_law PRIVATE LJMD_VERSION=1.0 _OMP_NAIVE)
