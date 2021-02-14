cmake_minimum_required(VERSION 3.10)


if(ENABLE_MPI)
	find_package(MPI REQUIRED) 
	
	#generic MPI test
	function(create_test_mpi)
		set(TEST_NPROCS 1)
		
		set(oneValueArgs NAME NPROCS)
		set(multiValueArgs SRC COMMAND LIBS)
		cmake_parse_arguments(TEST  "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
		
		set( TEST_MPI_NAME "test_${TEST_NAME}" )
		
		#message(" name ${TEST_NAME} nprocs ${TEST_NPROCS} src ${TEST_SRC} command ${TEST_COMMAND} libs ${TEST_LIBS}")
		#message("${TEST_NAME} command : ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NPROCS} ${MPIEXEC_PREFLAGS} ./${TEST_MPI_NAME} ${TEST_COMMAND}")
		
		add_executable(${TEST_MPI_NAME} ${TEST_SRC})
		set_target_properties(${TEST_MPI_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
		target_compile_definitions(${TEST_MPI_NAME} PRIVATE MPI_ENABLED=1 )
		target_include_directories(${TEST_MPI_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH} )
		target_link_libraries(${TEST_MPI_NAME} PRIVATE GTest::GTestMain GTest::GTest ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${TEST_LIBS} )
	
		add_test(
			NAME ${TEST_NAME} 
			COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NPROCS} ${TEST_COMMAND} ${TEST_NPROCS}
		)

	endfunction(create_test_mpi)
endif()




include_directories(${ljmd_SOURCE_DIR}/include)

set(ljmd_SOURCE_FILES ${ljmd_SOURCE_DIR}/src )

message("-- Compiling unit tests")



set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(GTest)

#common.c tests

add_executable(test_common ${ljmd_SOURCE_FILES}/common/test_common.cpp)
set_target_properties(test_common PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_common PRIVATE ${CMAKE_MODULE_PATH}/src)
target_link_libraries(test_common PRIVATE GTest::GTestMain GTest::GTest ljmd_common_lib)
add_test(NAME common COMMAND test_common )
  

# input tests
if(NOT ENABLE_MPI)

add_executable(test_input ${ljmd_SOURCE_FILES}/io/test_input.cpp)
set_target_properties(test_input PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_input PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/io ${ljmd_SOURCE_DIR}/src/common)
target_link_libraries(test_input PRIVATE GTest::GTestMain GTest::GTest ljmd_io_lib)
add_test(NAME input COMMAND test_input)

endif()


# output tests
add_executable(test_output ${ljmd_SOURCE_FILES}/io/test_output.cpp)
set_target_properties(test_output PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_output PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/io)
target_link_libraries(test_output PRIVATE GTest::GTestMain GTest::GTest ljmd_io_lib)
add_test(NAME output COMMAND test_output)

# utils tests

add_executable(test_utils ${ljmd_SOURCE_FILES}/physics/test_utils.cpp)
set_target_properties(test_utils PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_utils PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/physics)
target_link_libraries(test_utils PRIVATE GTest::GTestMain GTest::GTest ljmd_phys_lib)
add_test(NAME utils COMMAND test_utils )



if(NOT ENABLE_MPI)

# force tests

add_executable(test_force ${ljmd_SOURCE_FILES}/physics/test_force.cpp)
set_target_properties(test_force PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_force PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/physics)
target_link_libraries(test_force PRIVATE GTest::GTestMain GTest::GTest ljmd_phys_lib ljmd_common_lib)
add_test(NAME force COMMAND test_force )


endif()


# ekin and temp tests


add_executable(test_ek_t ${ljmd_SOURCE_FILES}/physics/test_ekin.cpp)
set_target_properties(test_ek_t PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_ek_t PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/physics)
target_link_libraries(test_ek_t PRIVATE GTest::GTestMain GTest::GTest ljmd_phys_lib ljmd_common_lib)
add_test(NAME ekin COMMAND test_ek_t )



# verlet integration tests

add_executable(test_verlet ${ljmd_SOURCE_FILES}/physics/test_integration.cpp)
set_target_properties(test_verlet PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_verlet PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/physics)
target_link_libraries(test_verlet PRIVATE GTest::GTestMain GTest::GTest ljmd_phys_lib ljmd_common_lib)
add_test(NAME verlet COMMAND test_verlet )


# reference test
find_program(BASH_PROGRAM bash)
find_program(AWK_PROGRAM awk)
find_program(CMP_PROGRAM cmp)
if(BASH_PROGRAM AND AWK_PROGRAM AND CMP_PROGRAM)
	
	if(ENABLE_MPI)
		list( APPEND comparison_test_args $<TARGET_FILE:mpi> ${CMAKE_MODULE_PATH} "mpi" )
	else()
		list( APPEND comparison_test_args $<TARGET_FILE:serial> ${CMAKE_MODULE_PATH} )
	endif()
    add_test(NAME comparison COMMAND ${ljmd_SOURCE_DIR}/tests/compare.sh ${comparison_test_args} )
endif()


#mpi tests

if(ENABLE_MPI)
	#Basic MPI test
	
	create_test_mpi(NAME mpi_basic NPROCS 4 SRC ${ljmd_SOURCE_FILES}/mpi/test_mpi_basic.cpp COMMAND $<TARGET_FILE:test_mpi_basic> LIBS )
	
	
	# MPI utils test
	
	add_executable(test_mpi_utils ${ljmd_SOURCE_FILES}/mpi/mpi_utils/test_mpi_utils.cpp)
	set_target_properties(test_mpi_utils PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
	target_include_directories(test_mpi_utils PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/mpi/mpi_utils)
	target_link_libraries(test_mpi_utils PRIVATE GTest::GTestMain GTest::GTest ljmd_mpi_util_lib m)
	add_test(NAME mpi_utils COMMAND test_mpi_utils )
	
	
	# MPI IO test
	#set( MPI_IO_TEST_LIBS 
	#		ljmd_mpi_util_lib
	#		ljmd_common_lib
	#	)
	#create_test_mpi(NAME mpi_io NPROCS 4 SRC ${ljmd_SOURCE_FILES}/mpi/mpi_io/test_mpi_io.cpp 
	#			COMMAND $<TARGET_FILE:test_mpi_io>  LIBS ${MPI_IO_TEST_LIBS}  )
	#configure_file(${ljmd_SOURCE_DIR}/src/mpi/mpi_io/test_params.inp ${ljmd_SOURCE_DIR}/build/test_params.inp COPYONLY)
	
	# MPI communication functions test
	set( MPI_COMM_TEST_LIBS 
			ljmd_mpi_util_lib
			ljmd_mpi_comm_lib
		)
	create_test_mpi(NAME mpi_comm NPROCS 4 SRC ${ljmd_SOURCE_FILES}/mpi/mpi_comm/test_mpi_comm.cpp 
				COMMAND $<TARGET_FILE:test_mpi_comm>  LIBS ${MPI_COMM_TEST_LIBS}  )
				
	###MPI ekin test
	##set( MPI_EKT_TEST_LIBS 
	##		ljmd_phys_lib
	##		ljmd_common_lib
	##	)
	##create_test_mpi(NAME ekin NPROCS 4 SRC ${ljmd_SOURCE_FILES}/physics/test_ekin.cpp
	##			COMMAND $<TARGET_FILE:test_ekin>  LIBS ${MPI_EKT_TEST_LIBS}  )
	##
	##		
	###MPI force test
	##set( MPI_FORCE_TEST_LIBS 
	##		ljmd_phys_lib
	##		ljmd_common_lib
	##	)
	##create_test_mpi(NAME force NPROCS 4 SRC ${ljmd_SOURCE_FILES}/physics/test_force.cpp
	##			COMMAND $<TARGET_FILE:test_force>  LIBS ${MPI_FORCE_TEST_LIBS}  )
	##	
	##	
	###MPI integration test
	##set( MPI_FORCE_TEST_LIBS 
	##		ljmd_phys_lib
	##		ljmd_common_lib
	##	)
	##create_test_mpi(NAME verlet NPROCS 4 SRC ${ljmd_SOURCE_FILES}/physics/test_integration.cpp
	##			COMMAND $<TARGET_FILE:test_verlet>  LIBS ${MPI_FORCE_TEST_LIBS}  )
	
	#MPI force test
	set( MPI_FORCE_TEST_LIBS 
			ljmd_phys_lib
			ljmd_common_lib
			ljmd_mpi_util_lib
		)
	create_test_mpi(NAME mpi_force NPROCS 3 SRC ${ljmd_SOURCE_FILES}/physics/test_mpi_force.cpp
				COMMAND $<TARGET_FILE:test_mpi_force>  LIBS ${MPI_FORCE_TEST_LIBS}  )
				
endif()


#################################
# All the OMP tests
#################################

# verlet integration with omp_naive
add_executable(test_verlet_omp_naive ${ljmd_SOURCE_FILES}/physics/test_integration.cpp)
set_target_properties(test_verlet_omp_naive PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_verlet_omp_naive PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/physics)
find_package(OpenMP REQUIRED)
target_link_libraries(test_verlet_omp_naive PRIVATE GTest::GTestMain GTest::GTest ljmd_phys_lib_omp_naive ljmd_common_lib OpenMP::OpenMP_CXX)
add_test(NAME verlet_omp_naive COMMAND test_verlet_omp_naive)

# force calculation with omp_naive
add_executable(test_force_omp_naive ${ljmd_SOURCE_FILES}/physics/test_force.cpp)
set_target_properties(test_force_omp_naive PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_force_omp_naive PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/physics)
find_package(OpenMP REQUIRED)
target_link_libraries(test_force_omp_naive PRIVATE GTest::GTestMain GTest::GTest ljmd_force_lib_omp_naive ljmd_common_lib OpenMP::OpenMP_CXX)
add_test(NAME force_omp_naive COMMAND test_force_omp_naive)

# ekin and temp with omp_naive
add_executable(test_ekin_omp_naive ${ljmd_SOURCE_FILES}/physics/test_ekin.cpp)
set_target_properties(test_ekin_omp_naive PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_ekin_omp_naive PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/physics)
find_package(OpenMP REQUIRED)
target_link_libraries(test_ekin_omp_naive PRIVATE GTest::GTestMain GTest::GTest ljmd_phys_lib_omp_naive ljmd_common_lib OpenMP::OpenMP_CXX)
add_test(NAME ekin_omp_naive COMMAND test_ekin_omp_naive)

# force calculation with omp_3rd_law
add_executable(test_force_omp_3rd_law ${ljmd_SOURCE_FILES}/physics/test_force.cpp)
set_target_properties(test_force_omp_3rd_law PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ljmd_SOURCE_DIR}/build)
target_include_directories(test_force_omp_3rd_law PRIVATE ${CMAKE_MODULE_PATH}/src ${ljmd_SOURCE_DIR}/src/physics)
find_package(OpenMP REQUIRED)
target_link_libraries(test_force_omp_3rd_law PRIVATE GTest::GTestMain GTest::GTest ljmd_force_lib_omp_3rd_law ljmd_common_lib OpenMP::OpenMP_CXX)
add_test(NAME force_omp_3rd_law COMMAND test_force_omp_3rd_law)

