#ifndef _HDF5_RESTART_H
#define _HDF5_RESTART_H

#define BUFF 512

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
} dataset_t;

int define_dataset( dataset_t *var, char *name, int ndims, int tsize, hsize_t *span );
int add_subset( dataset_t *var, void *ptr, hsize_t *offset, hsize_t *cnt );
int read_dataset( char *fn, dataset_t var );
int write_dataset( char *fn, dataset_t var );

#endif // _HDF5_RESTART_H
