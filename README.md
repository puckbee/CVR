# CVR
Parallelized and vectorized SpMV on Intel Xeon Phi (Knights Landing)
This is the artifact of our CGO'2018 paper [ CVR: Efficient SpMV Vectorization on X86 Processors ].

# Build

CVR can be built with a simple command 'make', while the resulted binariy file is 'spmv.cvr'.

# Data set Preparation
Our implementation of CVR supports sparse matrices with matrix market format, which is one of the default formats in SuiteSparse Matrix Collection (formerly the University of Florida Sparse Matrix Collection). Most of the data sets used in our paper can be found in this collection. We take web-Google for an example to demonstrate the data set preparation.

Step 1: wget https://sparse.tamu.edu/MM/SNAP/web-Google.tar.gz   [Note that we use the Matrix Market format]

Step 2: tar xvf web-Google.tar.gz

Then we get our sample data set: web-Google.mtx.

# Execution

Our CVR code accepts three parameters: file path; Number of Threads; Number of Iterations.

Sample:  ./spmv.cvr web-Google/web-Google.mtx 272 1000

It means CVR reads a sparse matrix from "web-Google/web-Google.mtx" and execute SpMV with 272 threads for 1000 iterations. 


