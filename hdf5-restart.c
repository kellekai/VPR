#include <hdf5.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "hdf5-restart.h"

/**
 * THIS FUNCTION CREATES A DATASET WITH PROPERTIES:
 * tsize -> TYPE SIZE OF A DATASET MEMBER
 * ndim  -> RANK (DIMENSIONALITY) OF THE DATASET
 */
int define_dataset( dataset_t *var, char *name, int ndims, int tsize, hsize_t *span, tmode_t mode ) 
{
    var->ndims      = ndims;
    var->tsize      = tsize;
    var->span       = (hsize_t*) calloc( ndims, sizeof(hsize_t) );
    memcpy( var->span, span, sizeof(hsize_t) * ndims );
    var->nparts     = 0;
    var->part       = NULL;
    strncpy( var->name, name, BUFF );
    var->mode       = mode;
}

/**
 * THIS FUNCTION ASSIGNS A CONTIGUOUS REGION (PART) 
 * TO THE DATASET
 */
int add_subset( dataset_t *var, void *ptr, hsize_t *offset, hsize_t *cnt )
{
    var->part = (data_t*) realloc( var->part, sizeof(data_t) * ++var->nparts );
    int idx = var->nparts-1;
    memset( &var->part[idx], 0x0, sizeof(data_t) );
    var->part[idx].ptr = ptr;
    if(offset) var->part[idx].offset = (hsize_t*) malloc( sizeof(hsize_t) * var->ndims );
    if(cnt) var->part[idx].count = (hsize_t*) malloc( sizeof(hsize_t) * var->ndims );
    memcpy( var->part[idx].offset, offset, sizeof(hsize_t) * var->ndims ); 
    memcpy( var->part[idx].count, cnt, sizeof(hsize_t) * var->ndims );
}

/**
 * THIS FUNCTION WRITES THE DATASET TO THE HDF5 FILE
 **/
int write_dataset( char *fn, dataset_t var ) {

    hid_t fid;      // file id
    hid_t tid;      // type id
    hid_t did;      // dataset id
    herr_t status;  // error status   

    // open hdf5 file using MPI
    fid = create_file( fn );
    
    // create derived datatype
    tid = H5Tcopy( H5T_NATIVE_INT );
    H5Tset_size( tid, var.tsize );

    // open the dataset
    switch( var.mode ) {
        case SHARD_DATA:
            // open dataset collectively
            did = create_shard_dataset( var, fid, tid );
            status = write_shard_dataset( var, did, tid );
            break;
        case GLOBL_DATA:
            did = create_globl_dataset( var, fid, tid );
            status = write_globl_dataset( var, did, tid );
            break;
    }
   
    H5Tclose(tid);
    H5Dclose(did);
    H5Fclose(fid);

}

/**
 * THIS FUNCTION LOADS THE DATASET FROM THE HDF5 FILE
 **/
int read_dataset( char *fn, dataset_t var )
{

    hid_t fid;      // file id
    hid_t tid;      // type id
    hid_t did;      // dataset id
    herr_t status;  // error status   

    // open hdf5 file using MPI
    fid = open_file( fn );
    
    // create derived datatype
    tid = H5Tcopy( H5T_NATIVE_INT );
    H5Tset_size( tid, var.tsize );

    // open the dataset
    switch( var.mode ) {
        case SHARD_DATA:
            // open dataset collectively
            did = open_shard_dataset( var, fid );
            status = read_shard_dataset( var, did, tid );
            break;
        case GLOBL_DATA:
            did = open_globl_dataset( var, fid );
            status = read_globl_dataset( var, did, tid );
            break;
    }
   
    H5Tclose(tid);
    H5Dclose(did);
    H5Fclose(fid);

}

herr_t read_shard_dataset( dataset_t var, hid_t did, hid_t tid ) 
{
    hid_t msid;     // memory space id
    hid_t fsid;     // file space id
    hid_t plid;     // property list id
    herr_t status;  // error status   
    
    status = 0;

    // create file space
    fsid = H5Screate_simple(var.ndims, var.span, NULL);
    
    // create property list for collective dataset write.
    plid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plid, H5FD_MPIO_COLLECTIVE);
   
    int i;
    for(i=0; i<var.nparts; ++i) {
        // select file region
        msid = H5Screate_simple(var.ndims, var.part[i].count, NULL);
        H5Sselect_hyperslab(fsid, H5S_SELECT_SET, var.part[i].offset, NULL, var.part[i].count, NULL);
        status += H5Dread(did, tid, msid, fsid, plid, var.part[i].ptr);
    }
    
    H5Sclose(fsid);
    H5Sclose(msid);
    H5Pclose(plid);

    return status;

}

herr_t read_globl_dataset( dataset_t var, hid_t did, hid_t tid ) 
{
    
    hid_t plid;     // property list id
    herr_t status;  // error status   
    
    // create property list for collective dataset write.
    plid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plid, H5FD_MPIO_COLLECTIVE);
    
    status = 0;

    int i;
    for(i=0; i<var.nparts; ++i) {
        status += H5Dread(did, tid, H5S_ALL, H5S_ALL, plid, var.part[i].ptr);
    }
    
    H5Pclose(plid);

    return status;

}

herr_t write_shard_dataset( dataset_t var, hid_t did, hid_t tid ) 
{
    hid_t msid;     // memory space id
    hid_t fsid;     // file space id
    hid_t plid;     // property list id
    herr_t status;  // error status   
    
    status = 0;

    // create file space
    fsid = H5Screate_simple(var.ndims, var.span, NULL);
    
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
    
    H5Sclose(fsid);
    H5Sclose(msid);
    H5Pclose(plid);

    return status;

}

herr_t write_globl_dataset( dataset_t var, hid_t did, hid_t tid ) 
{
    hid_t plid;     // property list id
    herr_t status;  // error status   
    
    status = 0;
    
    // create property list for collective dataset write.
    plid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plid, H5FD_MPIO_INDEPENDENT);
   
    int rank; MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if( !rank ) {
        int i;
        for(i=0; i<var.nparts; ++i) {
            status = H5Dwrite(did, tid, H5S_ALL, H5S_ALL, plid, var.part[i].ptr);
        }
    }

    H5Pclose(plid);

    return status;

}

int load_dataset_dims( char *fn, dataset_t *var ) 
{
    
    hid_t fid;
    hid_t did;
    hid_t sid;
    hid_t pid;
    
    fid = open_file( fn );

    did = open_shard_dataset( *var, fid );

    sid = H5Dget_space( did );

    var->ndims = H5Sget_simple_extent_ndims( sid );
    var->span = (hsize_t*) malloc( var->ndims * sizeof(hsize_t) );
    H5Sget_simple_extent_dims( sid, var->span, NULL );

    H5Pclose(pid);
    H5Sclose(sid);
    H5Dclose(did);
    H5Fclose(fid);

}

hid_t create_file( char *fn ) {

    hid_t fid;      // file id
    hid_t plid;     // property list id
    herr_t status;  // error status   

    // create a new file collectively and release property list identifier.
    plid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plid, MPI_COMM_WORLD, MPI_INFO_NULL);
    fid = H5Fcreate(fn, H5F_ACC_TRUNC, H5P_DEFAULT, plid);
    H5Pclose(plid);
    
    return fid;

}

hid_t open_file( char *fn ) {

    hid_t fid;      // file id
    hid_t plid;     // property list id
    herr_t status;  // error status   

    // open a file collectively and release property list identifier.
    plid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plid, MPI_COMM_WORLD, MPI_INFO_NULL);
    fid = H5Fopen(fn, H5F_ACC_RDWR, plid);
    H5Pclose(plid);
    
    return fid;

}

hid_t open_shard_dataset( dataset_t var, hid_t loc ) 
{
    
    hid_t did;      // dataset id
    herr_t status;  // error status   

    // create the dataset
    did = H5Dopen(loc, var.name, H5P_DEFAULT);
   
    return did;

}

hid_t open_globl_dataset( dataset_t var, hid_t loc ) 
{
    
    hid_t did;      // dataset id
    herr_t status;  // error status   

    // create the dataset
    did = H5Dopen(loc, var.name, H5P_DEFAULT);
   
    return did;

}

hid_t create_shard_dataset( dataset_t var, hid_t loc, hid_t tid ) 
{
    
    hid_t did;      // dataset id
    herr_t status;  // error status   
    hid_t fsid;     // file space id

    // create file space
    fsid = H5Screate_simple(var.ndims, var.span, NULL);
    
    // create the dataset
    did = H5Dcreate(loc, var.name, tid, fsid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
    return did;

}

hid_t create_globl_dataset( dataset_t var, hid_t loc, hid_t tid ) 
{
    
    hid_t did;      // dataset id
    herr_t status;  // error status   
    hid_t fsid;     // file space id

    // create file space
    fsid = H5Screate_simple(var.ndims, var.span, NULL);
    
    // create the dataset
    did = H5Dcreate(loc, var.name, tid, fsid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
    return did;

}
