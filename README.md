# CVR
Parallelized and vectorized SpMV on Intel Xeon Phi (Knights Landing). 
This is the artifact of our CGO'2018 paper [ CVR: Efficient SpMV Vectorization on X86 Processors ].

# Build

CVR can be built simply with 'make', while the resulted binariy file is 'spmv.cvr'.

Step 1: make

# Data set Preparation and Execution
Our implementation of CVR supports sparse matrices with matrix market format, which is one of the default formats in SuiteSparse Matrix Collection (formerly the University of Florida Sparse Matrix Collection). Most of the data sets used in our paper can be found in this collection. We take web-Google for an example to demonstrate the data set preparation.

Here, we use web-Google for example to show how to use CVR:

step 1: ./run_sample.sh

The CVR accepts three parameters: file path; Number of Threads; Number of Iterations.
In run_sample.sh, there is a command like this:

numactl --membind=1 ./spmv.cvr dataset/web-Google.mtx 68 1000

It means CVR reads a sparse matrix from "web-Google/web-Google.mtx" and execute SpMV with 272 threads for 1000 iterations. 

CVR will print two times in seconds: [Pre-processing time] and [SpMV Execution time].
[Pre-processing time] is the time of converting a sparse matrix with CSR format to CVR format.
[SpMV Execution time] is the average time of running 1000 iterations of SpMV with CVR format. Note that 1000 can be changed by changing "Number of Iterations"

# Compare CVR with other formats/solutions

Step 1: cd ./solutions_for_comparison

Step 2: ./build.sh        // build all formats/ solutions

Step 3: ./run_comparison.sh     // run all formats/solutions
(a)     ./run_comparison.sh | grep 'Pre-processing'      // get the Pre-processing time.
(b)     ./run_comparison.sh | grep 'SpMV Execution'      // get the SpMV execution time.
(c)     ./run_comparison.sh | grep 'Throughput'          // get the Throughput(GFlops).


# Cache Performance Profiling (Additional)

The L1 and L2 cache miss ratio can be obtained by running the following command: 

amplxe-cl -collect-with runsa -knob event-config=MEM_UOPS_RETIRED.L1_MISS_LOADS,MEM_UOPS.RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.L2_HIT_LOADS,MEM_UOPS_RETIRED.L2_MISS_LOADS,MEM_UOPS_RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.ALL_STORES,INST_RETIRED.ANY_P -r ./vtune_result/vtune_cvr/ numactl --membind=1 ./spmv.cvr web-Google/web-Google.mtx 272 1000

Note that the L2 cache miss ratio should be calculated using the data of function "spmv_compute_kernel" instead of the whole program.

