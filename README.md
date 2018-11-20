Whats this?
===

prototype of a restart mechanism, that uses HDF5 in order to restart with a different amount of processes.

API
===

```C 
int define_dataset( dataset_t *var, 
                    char *name,
                    int ndims,
                    int tsize, 
                    hsize_t *span,
                    tmode_t mode, 
                    dtype_t type )
```  
init global dataset attributes:
`*var`  : pointer to `dataset_t` dataset variable  
`*name` : name string  
`ndims` : number of dimensions, or rank  
`tsize` : type size of dataset elements  
`*span` : array of rank `ndims`. Global amount of elements in each dimension  
`mode`  : either `SHARD_DATA` or `GLOBL_DATA`.  
`type`  : either `VARSIZE_INTG`, `VARSIZE_DBLE` or `VARSIZE_CHAR`.  
  
Datasets defined with `SHARD_DATA`, hold a subset of a global dataset. That is, the rank keeps his fraction of the decomposed dataset.  
Datasets defined with `GLOBL_DATA`, should typically keep meta data, such as, iteration number, comm size, etc. All the ranks should have the same value on these datasets and the data is only written by the master rank. 
The `type` spezifies the base type of the dataset. `long`, `unsigned long`, etc are all `VARSIZE_INTG`, `float` and `double` are all `VARSIZE_DBLE`, etc.  
```C 
int add_subset( dataset_t *var, 
                void *ptr, 
                hsize_t *offset, 
                hsize_t *cnt )
```  
append a rank local contiguous subset to dataset:  
`*ptr`    : memory address of contiguous buffer  
`*offset` : array of rank `ndims`. Coordinates of the subset-origin in global dataset  
`*cnt`    : array of rank `ndims`. Local amount of elements in each dimension of subset  
```C
int write_datasets( dataset_t *var )
```
Checkpoint datasets `var[]`
```C
int read_datasets( dataset_t *var )
```
Recover datasets `var[]`


TODO (IMPLEMENTATION)
===

* implement function that allows to update dataset subsets
* implement function that allows to assign a dataset to a group

