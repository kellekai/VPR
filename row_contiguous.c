#include <hdf5.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "hdf5-restart.h"

/**
 * run with x mpi rank, where x is the cube of an integer.
 */

/**
 * This app writes integers from 1 -> x*y into a 2 dim
 * dataset inside a hdf5 file. The integers are ordered
 * as follows:
 *  
 *      |- x -|       
 * +---------------------------+
 * | 1  2  3  4  ... x-2 x-1 x | 
 * | x+1 x+2 x+3               | –      
 * | .                         | |
 * | .                         | y                   
 * | .                         | |                 
 * | (y-1)x+1 ... xy-2 xy-3 xy | _     
 * +---------------------------+
 *
 * Executing with arg: 1 writes the HDF5 file
 * Executing with arg: 0 reads from the file
 *
 * The grid is partitioned in the following way:
 *
 * +------------------------+
 * | 1  2  3  4 ... sqrt(n) | 
 * |                        | –      
 * | .                      | |
 * | .                      | y                   
 * | .                      | |                 
 * |   ...  ...           n | _     
 * +------------------------+
 *
 * thus in a quadratic decomposition.
 *
 * In this example, the dataset is not completely contiguous.
 * Only the rows of the 2 dim subsets are contiguous.
 *
 * The restart may happen with a different amount
 * of processes n, but, fdim0/sqrt(n) and fdim1/sqrt(n)
 * has to be an integer > 0
 */

#define X 1024
#define Y 1024

#define fdim0 X
#define fdim1 Y

#define fn "row-conti.h5"
#define dn "shared dataset"

int rank; 
int size; 
int i,j;    // loop iterators

// dataset info structure
int main( int argc, char **argv ) {

//
// -->> INIT AND DEFINITIONS
//
    
    MPI_Init(NULL,NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ldim0 = fdim0/((int)sqrt(size));
    int ldim1 = fdim1/((int)sqrt(size));

    // create row contiguous array
    int **data = (int**) malloc( sizeof(int*) * ldim0 );
    for(i=0; i<ldim0; ++i) {
        data[i] = (int*) malloc( sizeof(int) * ldim1 );
    }

//
// -->> INIT DATASET
//
    
    // declare FTI dataset (prototype)
    dataset_t var;
    dataset_t meta;

    // set dataset properties
    hsize_t fdim[2] = { fdim0, fdim1 }; 
    
    // define shared dataset (the decomposed global data)
    define_dataset( &var, dn, 2, sizeof(int), fdim, SHARD_DATA );
    
    // define global dataset (meta data, e.g., iteration number, number of processes, etc.)
    hsize_t scalar[] = { 1 };
    define_dataset( &meta, "number of processes", 1, sizeof(int), scalar, GLOBL_DATA );
    add_subset( &meta, &size, NULL, NULL );

    // add sub regions to dataset (contiguous rows)
    hsize_t offset[] = { ((rank/((int)sqrt(size)))%((int)sqrt(size)))*ldim0, (rank%((int)sqrt(size)))*ldim1 };
    hsize_t count[] = { 1, ldim1 };
    
    int base = offset[0]*fdim[1] + offset[1] + 1;
    int stride = 0;
    // init array elements with increasing integers
    for(i=0; i<ldim0; ++i) {
        for(j=0; j<ldim1; ++j) {
            data[i][j] = base++;
        }
        base += fdim[1] - ldim1;
    }

    for(i=0; i<ldim0; ++i) {
        add_subset( &var, data[i], offset, count );
        offset[0]++;
    }
    

//
// -->> CHECKPOINT AND RESTART
//
 
    // if run with parameter 1 -> CHECKPOINT
    if( (argc > 1) && atoi(argv[1]) ) {
        // TODO rewrite this function so that it writes more then one dataset.
        write_dataset( fn, var );
    // if run with parameter 0 -> RESTART
    } else if( (argc > 1) && !atoi(argv[1]) ) {
        // reset data to -1
        for(i=0; i<ldim0; ++i) {
            for(j=0; j<ldim1; ++j) {
                data[i][j] = -1;
            }
        }
        read_dataset( fn, var );
        int out = 0;
        for(i=0; i<ldim0; ++i) {
            for(j=0; j<ldim1; ++j) {
                out += data[i][j];
            }
        }
        int res;
        int check = fdim[0]*fdim[1]*(fdim[0]*fdim[1]+1)/2;
        MPI_Allreduce( &out, &res, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ); 
        if(rank==0) 
            printf("[%d]: %s (res:%d,check:%d)\n", rank, (res == check) ? "SUCCESS" : "FAILURE", res, check ); 
    } else {
        if(rank==0) printf("no argument passed (value 1:checkpoint or 0:restart)\n");
    }

//
// -->> FINALIZE
//

    for(i=0; i<ldim0; ++i) {
        free(data[i]);
    }

    MPI_Finalize();
    
    return 0;

}
