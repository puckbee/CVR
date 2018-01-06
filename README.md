# CVR
Parallelized and vectorized SpMV on Intel Xeon Phi (Knights Landing). <br>
This is the artifact of our CGO'2018 paper [ CVR: Efficient Vectorization of SpMV on X86 Processors ]. <br>
You can find a copy of the paper in this repository named 'CVR-Efficient Vectorization of SpMV on X86 Processors.pdf'

# Build
CVR can be built simply with 'make', while the resulted binariy file is 'spmv.cvr'.

	Step: make       

# Dataset Preparation and Execution
Our implementation of CVR supports sparse matrices with matrix market format, which is one of the default formats in SuiteSparse Matrix Collection. Most of the data sets used in our paper can be found in either of these two collections:

1) [SuiteSparse Matrix Collection](https://sparse.tamu.edu) (formerly the University of Florida Sparse Matrix Collection).
2) [Stanford Large Network Dataset Collection](http://snap.stanford.edu/data/) (SNAP).

Here, we use web-Google for example to show how to use CVR:

	step 1: ./run_sample.sh

The CVR accepts three parameters: file path; Number of Threads; Number of Iterations. <br>
In run_sample.sh, there is a command like this:

**numactl --membind=1 ./spmv.cvr [filepath] [numThreads] [numIterations]**

**Sample: numactl --membind=1 ./spmv.cvr dataset/web-Google.mtx 68 1000**

CVR will print two times in seconds: [Pre-processing time] and [SpMV Execution time]. <br>
[Pre-processing time] is the time of converting a sparse matrix with CSR format to CVR format. <br>
[SpMV Execution time] is the average time of running 1000 iterations of SpMV with CVR format. Note that 1000 can be changed by changing "Number of Iterations" <br>

# Compare CVR with Other Formats/Solutions
MKL,CSR-I and ESB are dependent on MKL. <br>
Please make sure that MKL is already installed and the environment variable $MKL_ROOT is already set. <br>

We tried various threads numbers and parameters for each format/solution and choose the configuration that achieves the best performance.<br>
You can try to setup different threads numbers in run_comparison.sh, we will elaborate how to do this later. <br>
But if you just want to have a reproduce the experiment results of web-Google, these three steps can definitely meet your need. <br>

	Step 1: cd ./solutions_for_comparison

	Step 2: ./build.sh        // build all formats/ solutions

	Step 3: ./run_comparison.sh ../dataset/web-Google.mtx                           // run all formats/solutions 
	(a)     ./run_comparison.sh ../dataset/web-Google.mtx  | grep 'Pre-processing'  // get the Pre-processing time. 
	(b)     ./run_comparison.sh ../dataset/web-Google.mtx  | grep 'SpMV Execution'  // get the SpMV execution time. 
	(c)     ./run_comparison.sh ../dataset/web-Google.mtx  | grep 'Throughput'      // get the Throughput(GFlops).

We will elaborate how to use each format/solution, so that you can change the configuration to fullfill your own requirements.
### CSR5
**numactl --membind=1 ./bin/spmv.csr5 [filepath] [numThreads] [numIterations]**

**Sample: numactl --membind=1 ./spmv.csr5 ../dataset/web-Google.mtx 204 1000**
### VHCC
VHCC has many parameters. Since the width and height of blocks is pretty fixed to be (512,8192), we only provide the number of panels here.

**numactl --membind=1 ./bin/spmv.vhcc [filepath] [numThreads] [numIterations] [numPanels]**
		
**Sample: numactl --membind=1 ./spmv.vhcc ../dataset/web-Google.mtx 272 1000 1**
### CSR-I
**numactl --membind=1 ./bin/spmv.csr [filepath] [numThreads] [numIterations]**
	
**Sample: numactl --membind=1 ./spmv.csr ../dataset/web-Google.mtx 272 1000**
### ESB
ESB has diffent schedule policies: static and dynamic. 1 for static; 2 for dynamic; 3 for both two.<br>

**numactl --membind=1 ./bin/spmv.esb [filepath] [numThreads] [numIterations] [schedule_policy]**

**Sample: numactl --membind=1 ./spmv.esb ../dataset/web-Google.mtx 272 1000 3**
### MKL
**numactl --membind=1 ./bin/spmv.mkl [filepath] [numThreads] [numIterations]**

**Sample: numactl --membind=1 ./spmv.mkl ../dataset/web-Google.mtx 272 1000**


# Cache Performance Profiling (Additional)
Dependency:  Vtune

	Step 1: cd ./solutions_for_comparison
		
	Step 2: ./build.sh                 // If it has not been built yet

	Step 3: ./run_locality.sh [filepath][nT_CVR][nT_CSR5][nT_VHCC][nPanels][nT_CSRI][nT_ESB][schedule_ESB][nT_MKL]
	        ./run_locality.sh ../dataset/web-Google.mtx 68 204 272 1 272 272 1 272

	Note that 'nT' stands for numThreads, while 'nPanels' stands for numPanels of VHCC.

# Notes
We only modified the source code of CSR5 and VHCC to format the output messages. <br>
The source code of CSR5[ICS'15], please refer to (https://github.com/bhSPARSE/Benchmark_SpMV_using_CSR5)<br>
The source code of VHCC[CGO'15], please refer to (https://github.com/vhccspmv/vhcc) <br>

We provide the execution files instead of source codes for ESB, CSR-I and MKL. Please email me (xiebiwei at ict.ac.cn) if you want to review the codes. I can't put the codes on github, since I'm not sure about its license. Please refer to [MKL Sparse Package](https://software.intel.com/en-us/articles/intel-math-kernel-library-inspector-executor-sparse-blas-routines) for more informations.




