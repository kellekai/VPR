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
int define_dataset( dataset_t *var, char *name, int ndims, int tsize, hsize_t *span ) 
{
    var->ndims      = ndims;
    var->tsize      = tsize;
    var->span       = (hsize_t*) calloc( ndims, sizeof(hsize_t) );
    memcpy( var->span, span, sizeof(hsize_t) * ndims );
    var->nparts     = 0;
    var->part       = NULL;
    strncpy( var->name, name, BUFF );
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
int write_dataset( char *fn, dataset_t var ) {

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
    did = H5Dcreate(fid, var.name, tid, fsid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
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
int read_dataset( char *fn, dataset_t var )
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
    did = H5Dopen(fid, var.name, H5P_DEFAULT);
   
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

