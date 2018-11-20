CFLAGS := -g

all: check

check: main
	mpirun -n 4 ./main 1
	mpirun -n 64 ./main 0

main: row_contiguous.c hdf5-restart.o
	mpicc $(CFLAGS) -o $@ $< hdf5-restart.o -lhdf5 -lm

hdf5-restart.o: hdf5-restart.c hdf5-restart.h
	mpicc $(CFLAGS) -c $<

clean:
	rm -rf main row-conti.h5 hdf5-restart.o

