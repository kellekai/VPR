all: check

check: main
	mpirun -n 4 ./main 1
	mpirun -n 64 ./main 0

main: row_contiguous.c
	mpicc -o $@ $< -lhdf5 -lm

clean:
	rm -rf main row-conti.h5

