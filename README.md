Whats this?
===

prototype of a restart mechanism, that uses HDF5 in order to restart with a different amount of processes.

API
===

```C 
int define_dataset( dataset_t *var, int ndims, int tsize, hsize_t *span )
```  
init global dataset attributes:  
`ndims` : number of dimensions, or rank  
`tsize` : type size of dataset elements  
`*span` : array of rank `ndims`. Global amount of elements in each dimension  
```C 
int add_subset( dataset_t *var, void *ptr, hsize_t *offset, hsize_t *cnt )
```  
append a rank local contiguous subset to dataset:  
`*ptr`    : memory address of contiguous buffer  
`*offset` : array of rank `ndims`. Coordinates of the subset-origin in global dataset  
`*cnt`    : array of rank `ndims`. Local amount of elements in each dimension of subset  
```C
int write_dataset( dataset_t var )
```
Checkpoint dataset `var`
```C
int read_dataset( dataset_t var )
```
Recover dataset `var`


TODO (PROTOTYPE)
===

* implement function, that allows the recovery of the global dataset dimensions before FTI_Recovery
* implement example for a completely contiguous 2-dim dataset 
* implement example that handles a cell with residuals
* implement example that includes ghost cells

TODO (IMPLEMENTATION)
===

* implement function that allows to update dataset subsets
* implement function that allows to assign a dataset to a group

