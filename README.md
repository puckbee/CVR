# CVR
Parallelized and vectorized SpMV on Intel Xeon Phi (Knights Landing). 
This is the artifact of our CGO'2018 paper [ CVR: Efficient SpMV Vectorization on X86 Processors ].

# Build

CVR can be built simply with 'make', while the resulted binariy file is 'spmv.cvr'.

Step: make

# Data set Preparation
Our implementation of CVR supports sparse matrices with matrix market format, which is one of the default formats in SuiteSparse Matrix Collection (formerly the University of Florida Sparse Matrix Collection). Most of the data sets used in our paper can be found in this collection. We take web-Google for an example to demonstrate the data set preparation.

Step 1: wget https://sparse.tamu.edu/MM/SNAP/web-Google.tar.gz   [Note that we use the Matrix Market format]

Step 2: tar xvf web-Google.tar.gz

Then we get our sample data set: web-Google.mtx.

# Execution

Our CVR code accepts three parameters: file path; Number of Threads; Number of Iterations.

Step: numactl --membind=1 ./spmv.cvr web-Google/web-Google.mtx 272 1000

It means CVR reads a sparse matrix from "web-Google/web-Google.mtx" and execute SpMV with 272 threads for 1000 iterations. 

CVR will print two times in seconds: [pre-processing time] and [SpMV execution time].
[pre-processing time] is the time of converting a sparse matrix with CSR format to CVR format.
[SpMV execution time] is the average time of running 1000 iterations of SpMV with CVR format. Note that 1000 can be changed by changing "Number of Iterations"

# Cache Performance Profiling (Additional)

The L1 and L2 cache miss ratio can be obtained by running the following command: 

amplxe-cl -collect-with runsa -knob event-config=MEM_UOPS_RETIRED.L1_MISS_LOADS,MEM_UOPS.RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.L2_HIT_LOADS,MEM_UOPS_RETIRED.L2_MISS_LOADS,MEM_UOPS_RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.ALL_STORES,INST_RETIRED.ANY_P -r ./vtune_result/vtune_cvr/ numactl --membind=1 ./spmv.cvr web-Google/web-Google.mtx 272 1000

Note that the L2 cache miss ratio should be calculated using the data of function "spmv_compute_kernel" instead of the whole program.

