#ifndef _HDF5_RESTART_H
#define _HDF5_RESTART_H

#define BUFF 512

typedef enum dtype_t {
    VARSIZE_INTG,
    VARSIZE_DBLE,
    VARSIZE_CHAR
} dtype_t;

typedef enum tmode_t {
    SHARD_DATA,         // dataset shared by processes (decomposed)
    GLOBL_DATA,         // data that has the same value along all processes
    LOCAL_DATA          // data that is local to a process
} tmode_t;


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
    char name[BUFF];    // dataset name
    tmode_t mode;
    hid_t type;
} dataset_t;

int define_dataset( dataset_t *var, char *name, int ndims, int tsize, hsize_t *span, tmode_t mode, dtype_t type );
int add_subset( dataset_t *var, void *ptr, hsize_t *offset, hsize_t *cnt );
int read_datasets( char *fn, dataset_t *var, int num );
int write_datasets( char *fn, dataset_t *var, int num );
int load_dataset_dims( char *fn, dataset_t *var ); 

// helper routines
hid_t open_file( char *fn );
hid_t open_shard_dataset( dataset_t var, hid_t loc ); 
hid_t open_globl_dataset( dataset_t var, hid_t loc ); 
hid_t create_file( char *fn );
hid_t create_shard_dataset( dataset_t var, hid_t loc, hid_t tid ); 
hid_t create_globl_dataset( dataset_t var, hid_t loc, hid_t tid ); 
herr_t read_shard_dataset( dataset_t var, hid_t did, hid_t tid ); 
herr_t read_globl_dataset( dataset_t var, hid_t did, hid_t tid ); 
herr_t write_shard_dataset( dataset_t var, hid_t did, hid_t tid ); 
herr_t write_globl_dataset( dataset_t var, hid_t did, hid_t tid ); 
#endif // _HDF5_RESTART_H
