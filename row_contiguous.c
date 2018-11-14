#include <hdf5.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

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
typedef struct data_t {
    void *ptr;          // ptr to data
    hsize_t *offset;    // offset in global dataset grid per dimension
    hsize_t *count;     // local number of elements in subset per dimension
} data_t;

typedef struct dataset_t {
    int tsize;          // type size
    int ndims;          // dimensionality of dataset
    hsize_t *span;      // global number of elements in each dimension
    int nparts;         // number of subsets (contiguous regions)
    data_t *part;       // subsets
} dataset_t;

int define_dataset( dataset_t *var, int ndims, int tsize, hsize_t *span ); 
int add_subset( dataset_t *var, void *ptr, hsize_t *offset, hsize_t *cnt );
int read_dataset( dataset_t var );
int write_dataset( dataset_t var );

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

    // set dataset properties
    hsize_t fdim[2] = { fdim0, fdim1 }; 
    define_dataset( &var, 2, sizeof(int), fdim );

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
        write_dataset( var );
    // if run with parameter 0 -> RESTART
    } else if( (argc > 1) && !atoi(argv[1]) ) {
        // reset data to -1
        for(i=0; i<ldim0; ++i) {
            for(j=0; j<ldim1; ++j) {
                data[i][j] = -1;
            }
        }
        read_dataset( var );
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

/**
 * THIS FUNCTION CREATES A DATASET WITH PROPERTIES:
 * tsize -> TYPE SIZE OF A DATASET MEMBER
 * ndim  -> RANK (DIMENSIONALITY) OF THE DATASET
 */
int define_dataset( dataset_t *var, int ndims, int tsize, hsize_t *span ) 
{
    var->ndims      = ndims;
    var->tsize      = tsize;
    var->span       = (hsize_t*) calloc( ndims, sizeof(hsize_t) );
    memcpy( var->span, span, sizeof(hsize_t) * ndims );
    var->nparts     = 0;
    var->part       = NULL;
}

/**
 * THIS FUNCTION ASSIGNS A CONTIGUOUS REGION (PART) 
 * TO THE DATASET
 */
int add_subset( dataset_t *var, void *ptr, hsize_t *offset, hsize_t *cnt )
{
    var->part = (data_t*) realloc( var->part, sizeof(data_t) * ++var->nparts );
    int idx = var->nparts-1;
    var->part[idx].ptr = ptr;
    var->part[idx].offset = (hsize_t*) malloc( sizeof(hsize_t) * var->ndims );
    var->part[idx].count = (hsize_t*) malloc( sizeof(hsize_t) * var->ndims );
    memcpy( var->part[idx].offset, offset, sizeof(hsize_t) * var->ndims ); 
    memcpy( var->part[idx].count, cnt, sizeof(hsize_t) * var->ndims );
}

/**
 * THIS FUNCTION WRITES THE DATASET TO THE HDF5 FILE
 **/
int write_dataset( dataset_t var ) {

    hid_t fid;      // file id
    hid_t tid;      // type id
    hid_t did;      // dataset id
    hid_t msid;     // memory space id
    hid_t fsid;     // file space id
    hid_t plid;     // property list id
    herr_t status;  // error status   

    // create file space
    fsid = H5Screate_simple(var.ndims, var.span, NULL);

    // create a new file collectively and release property list identifier.
    plid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plid, MPI_COMM_WORLD, MPI_INFO_NULL);
    fid = H5Fcreate(fn, H5F_ACC_TRUNC, H5P_DEFAULT, plid);
    H5Pclose(plid);
    
    // create derived datatype
    tid = H5Tcopy( H5T_NATIVE_INT );
    H5Tset_size( tid, var.tsize );

    // create the dataset
    did = H5Dcreate(fid, dn, tid, fsid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
    // create property list for collective dataset write.
    plid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plid, H5FD_MPIO_COLLECTIVE);
   
    int i;
    for(i=0; i<var.nparts; ++i) {
        // select file region
        msid = H5Screate_simple(var.ndims, var.part[i].count, NULL);
        H5Sselect_hyperslab(fsid, H5S_SELECT_SET, var.part[i].offset, NULL, var.part[i].count, NULL);
        status = H5Dwrite(did, tid, msid, fsid, plid, var.part[i].ptr);
    }

    H5Tclose(tid);
    H5Pclose(plid);
    H5Sclose(fsid);
    H5Sclose(msid);
    H5Dclose(did);
    H5Fclose(fid);

}

/**
 * THIS FUNCTION LOADS THE DATASET FROM THE HDF5 FILE
 **/
int read_dataset( dataset_t var )
{

    hid_t fid;      // file id
    hid_t tid;      // type id
    hid_t did;      // dataset id
    hid_t msid;     // memory space id
    hid_t fsid;     // file space id
    hid_t plid;     // property list id
    herr_t status;  // error status   

    // create file space
    fsid = H5Screate_simple(var.ndims, var.span, NULL);

    // create a new file collectively and release property list identifier.
    plid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plid, MPI_COMM_WORLD, MPI_INFO_NULL);
    fid = H5Fopen(fn, H5F_ACC_RDWR, plid);
    H5Pclose(plid);
    
    // create derived datatype
    tid = H5Tcopy( H5T_NATIVE_INT );
    H5Tset_size( tid, var.tsize );

    // create the dataset
    did = H5Dopen(fid, dn, H5P_DEFAULT);
   
    // create property list for collective dataset write.
    plid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plid, H5FD_MPIO_COLLECTIVE);
   
    int i;
    for(i=0; i<var.nparts; ++i) {
        // select file region
        msid = H5Screate_simple(var.ndims, var.part[i].count, NULL);
        H5Sselect_hyperslab(fsid, H5S_SELECT_SET, var.part[i].offset, NULL, var.part[i].count, NULL);
        status = H5Dread(did, tid, msid, fsid, plid, var.part[i].ptr);
    }

    H5Tclose(tid);
    H5Pclose(plid);
    H5Sclose(fsid);
    H5Sclose(msid);
    H5Dclose(did);
    H5Fclose(fid);

}
