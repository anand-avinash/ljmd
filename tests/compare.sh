#!/bin/bash


run_test() {
	INPUT_DIR="$1"
	shift
	CMD="$@"
	
	#echo "test $CMD $INPUT_DIR"
	
	${INPUT_DIR}/input_files/argon_108.sh ${INPUT_DIR}/input_files | ${CMD}	

	head -10 argon_108.dat | awk '{printf("%d %.6f %.6f %.6f\n",$1,$2,$3,$4);}'> a.dat
	head -10 ${INPUT_DIR}/references/argon_108.dat | awk '{printf("%d %.6f %.6f %.6f\n",$1,$2,$3,$4);}'> b.dat
	SUCCESS=0
	cmp a.dat b.dat && SUCCESS=1

	rm -f a.dat b.dat argon_108.dat

	if [ ${SUCCESS} -ne 1 ]
	then
		exit 1
	fi

}

BINARY=$1
INPUT_DIR=$2
SWITCH=$3

#echo "comparison test args : ${BINARY} ${INPUT_DIR} ${SWITCH}  "




if [ "$SWITCH" = "mpi" ]; then
		
		for NPROCS in 1 4; do
				echo "parallel test with ${NPROCS} processors"
				run_test "${INPUT_DIR}" "mpirun -np ${NPROCS} ${BINARY}"
		done

fi

if [ "${SWITCH}" = "serial" ]; then
		echo "serial test"
		run_test "${INPUT_DIR}" "${BINARY}"
fi

if [ "${SWITCH}" = "omp_naive" ]; then
	echo "omp_naive test with default number of threads"
	run_test "${INPUT_DIR}" "${BINARY}"
fi

if [ "${SWITCH}" = "omp_3rd_law" ]; then
	echo "omp_3rd_law test with default number of threads"
	run_test "${INPUT_DIR}" "${BINARY}"
fi