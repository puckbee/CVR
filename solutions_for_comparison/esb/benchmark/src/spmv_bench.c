/*******************************************************************************
*   Copyright(C) 2013 Intel Corporation. All Rights Reserved.
*
*   The source code, information  and  material ("Material") contained herein is
*   owned  by Intel Corporation or its suppliers or licensors, and title to such
*   Material remains  with Intel Corporation  or its suppliers or licensors. The
*   Material  contains proprietary information  of  Intel or  its  suppliers and
*   licensors. The  Material is protected by worldwide copyright laws and treaty
*   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
*   modified, published, uploaded, posted, transmitted, distributed or disclosed
*   in any way  without Intel's  prior  express written  permission. No  license
*   under  any patent, copyright  or  other intellectual property rights  in the
*   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
*   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
*   intellectual  property  rights must  be express  and  approved  by  Intel in
*   writing.
*
*   *Third Party trademarks are the property of their respective owners.
*
*   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
*   this  notice or  any other notice embedded  in Materials by Intel or Intel's
*   suppliers or licensors in any way.
*
********************************************************************************
*
*   Content :  Double-precision performance benchmark for
*              Intel(R) MKL SpMV Format Prototype Package, version 0.2
*
*   usage:  ./exec.file sparse_matrix_in_matrix_market_format
*
********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "mkl.h"
#include <math.h>

#include <unistd.h>
#include <sys/mman.h>

#ifdef __KNC__
    #include "spmv_interface.h"
#endif

// alignment for memory allocation in mkl_malloc
#define ALIGN 512
// threshold for validation of SpMV results 
#define EPS 1.0e-15

// This macro simplifies error handling in main() routine
#define __CHECK_ERROR_AND_CALL__( RES, GOOD, MESSAGE ) { \
    if ( !error && GOOD != RES ) \
    { \
        fprintf(stderr, MESSAGE ); \
        error = -1; \
    } \
}

// This structure is used for storing sparse matrices in COO and CSR formats
struct SparseMatrix
{
    int num_rows;   // number of rows in the matrix
    int num_cols;   // number of columns in the matrix
    int nnz;        // number of non-zero elements in the matrix
    int *rows;      // array with row indices
    int *cols;      // array with column indices
    double *vals;   // array with non-zero elements
};

static int  readSparseCOOMatrix ( FILE *f, struct SparseMatrix *cooMatrix );
static int  convertCOO2CSR ( const struct SparseMatrix *cooMatrix, struct SparseMatrix *csrMatrix );
static void deleteSparseMatrix ( struct SparseMatrix *matrix );

static void printMatrixInfo ( const struct SparseMatrix *csrMatrix );
static double calcFrobeniusNorm ( int vectorLength, double *vectorValues );

static int  benchmark_MKL_SpMV ( const struct SparseMatrix *csrMatrix, double alpha, double *x,
                                 double beta, double *y, double *y_ref, double mflop, double matrixFrobeniusNorm, char* filename, int numIterations, int numThreads );

#ifdef __KNC__
static int  benchmark_CSR_SpMV ( const struct SparseMatrix *csrMatrix, double alpha, double *x,
                                 double beta, double *y, double *y_ref, double mflop, double matrixFrobeniusNorm, 
                                 sparseSchedule_t schedule, char* filename, int numIterations, int numThreads );
static int  benchmark_ESB_SpMV ( const struct SparseMatrix *csrMatrix, double alpha, double *x,
                                 double beta, double *y, double *y_ref, double mflop, double matrixFrobeniusNorm, 
                                 sparseSchedule_t schedule, char* filename, int numIterations, int numThreads );
#endif

int main( int argc, char *argv[] )
{
    FILE   *f;                      // input file with sparse matrix data in matrix-market format
    struct SparseMatrix cooMatrix,  // input matrix in COO format
    csrMatrix;                  // converted matrix in CSR format
    double  *y, *y_ref;             // output vectors of length num_rows
    double  *x;                     // input vector  of length num_cols
    double alpha, beta;             // SpMV parameters
    double mflop;                   // amount of work in SpMV routines in MFlops
    int error = 0;                   
    double matrixFrobeniusNorm = 0.0;

    printf( "Double-precision SpMV performance benchmark for \n Intel(R) MKL SpMV Format Prototype Package, version 0.2\n" );

    if(atoi(argv[2]) == 0)
      return 0;

    omp_set_num_threads(atoi(argv[2]));

    int numThreads = atoi(argv[2]);
    char* filename = argv[1];

    if ( argc < 2 )
    {
        fprintf( stderr, "Insufficient number of input parameters:\n");
        fprintf( stderr, "File name with sparse matrix for benchmarking is missing\n" );
        fprintf( stderr, "Usage: %s [martix market filename]\n", argv[0] );
        exit(1);
    }
    else
    {
        if ( (f = fopen(argv[1], "r")) == NULL )
        {
            fprintf( stderr, "Error opening input file: %s\n", argv[1] );
            exit(1);
        }
        else
        {
            printf( "Input file: %s\n", argv[1] );
        }
    }

    if ( 0 != readSparseCOOMatrix( f, &cooMatrix ) )
    {
        fprintf(stderr, "Reading COO matrix in matrix market format failed\n" );
        return -2;
    }

    __CHECK_ERROR_AND_CALL__ ( convertCOO2CSR( &cooMatrix, &csrMatrix ),
                               0, "Conversion of matrices from COO to CSR failed\n" )

    deleteSparseMatrix( &cooMatrix );

    if ( error )
        return error;

//    mflop = 2.0 * csrMatrix.nnz / 1.0e6; // Amount of SpMV work for time estimation;
    mflop = csrMatrix.nnz / 1.0e6; // Amount of SpMV work for time estimation;

    // Align allocated memory to boost performance 
#ifdef MMAP
    x     = (double*)mmap(0, csrMatrix.num_cols* sizeof(double), PROT_READ|PROT_WRITE,MAP_ANONYMOUS|MAP_PRIVATE|MAP_HUGETLB,-1,0);
#else
    x     = ( double* ) MKL_malloc ( csrMatrix.num_cols * sizeof( double ), ALIGN );
#endif
    y     = ( double* ) MKL_malloc ( csrMatrix.num_rows * sizeof( double ), ALIGN );
    y_ref = ( double* ) MKL_malloc ( csrMatrix.num_rows * sizeof( double ), ALIGN );

    if ( NULL == x || NULL == y || NULL == y_ref )
    {
        fprintf( stderr, "Could not allocate memory for vectors!\n" );
//        MKL_free( x );
        MKL_free( y );
        MKL_free( y_ref );

        deleteSparseMatrix( &csrMatrix );
        return -1;
    }

//    alpha =  M_PI / 2;  // M_PI = 3.14159265358979323846 in <math.h>
//    beta  = -M_PI / 3;

    alpha = 1;
    beta = 1;
 
    matrixFrobeniusNorm = calcFrobeniusNorm ( csrMatrix.nnz, csrMatrix.vals );

    int numIterations = atoi(argv[3]);

    printMatrixInfo( &csrMatrix );

#ifdef _SpMV_MKL_
    __CHECK_ERROR_AND_CALL__ ( benchmark_MKL_SpMV ( &csrMatrix, alpha, x, beta, y, y_ref, mflop, matrixFrobeniusNorm , filename, numIterations, numThreads),
                               0, "Benchmarking MKL SpMV failed\n" )
#endif

#ifdef __KNC__

#ifdef _SpMV_CSR_
    __CHECK_ERROR_AND_CALL__ ( benchmark_CSR_SpMV ( &csrMatrix, alpha, x, beta, y, y_ref, mflop, matrixFrobeniusNorm, 
                                                    INTEL_SPARSE_SCHEDULE_STATIC , filename, numIterations, numThreads),
                               0, "Benchmarking CSR SpMV with static scheduling failed\n" )
#endif
//    __CHECK_ERROR_AND_CALL__ ( benchmark_CSR_SpMV ( &csrMatrix, alpha, x, beta, y, y_ref, mflop, matrixFrobeniusNorm, 
//                                                    INTEL_SPARSE_SCHEDULE_DYNAMIC , filename),
//                               0, "Benchmarking CSR SpMV with dynamic scheduling failed\n" )

//    __CHECK_ERROR_AND_CALL__ ( benchmark_CSR_SpMV ( &csrMatrix, alpha, x, beta, y, y_ref, mflop, matrixFrobeniusNorm, 
//                                                    INTEL_SPARSE_SCHEDULE_BLOCK , filename),
// 
//                              0, "Benchmarking CSR SpMV with block scheduling failed\n" )
#ifdef _SpMV_ESB_
  if(atoi(argv[4]) == 1)
  {
    __CHECK_ERROR_AND_CALL__ ( benchmark_ESB_SpMV ( &csrMatrix, alpha, x, beta, y, y_ref, mflop, matrixFrobeniusNorm, 
                                                    INTEL_SPARSE_SCHEDULE_STATIC, filename, numIterations, numThreads ),
                               0, "Benchmarking ESB SpMV with static scheduling failed\n" )
  }
  else if(atoi(argv[4]) == 2)
  {
    __CHECK_ERROR_AND_CALL__ ( benchmark_ESB_SpMV ( &csrMatrix, alpha, x, beta, y, y_ref, mflop, matrixFrobeniusNorm, 
                                                    INTEL_SPARSE_SCHEDULE_DYNAMIC , filename, numIterations, numThreads ),
                               0, "Benchmarking ESB SpMV with dynamic scheduling failed\n" )
  }
  else if(atoi(argv[4]) == 3)
  {

    __CHECK_ERROR_AND_CALL__ ( benchmark_ESB_SpMV ( &csrMatrix, alpha, x, beta, y, y_ref, mflop, matrixFrobeniusNorm, 
                                                    INTEL_SPARSE_SCHEDULE_STATIC, filename, numIterations, numThreads ),
                               0, "Benchmarking ESB SpMV with static scheduling failed\n" )
    __CHECK_ERROR_AND_CALL__ ( benchmark_ESB_SpMV ( &csrMatrix, alpha, x, beta, y, y_ref, mflop, matrixFrobeniusNorm, 
                                                    INTEL_SPARSE_SCHEDULE_DYNAMIC , filename, numIterations, numThreads ),
                               0, "Benchmarking ESB SpMV with dynamic scheduling failed\n" )
  }
  
#endif
#endif

//    MKL_free( x );
    MKL_free( y );
    MKL_free( y_ref );

    deleteSparseMatrix( &csrMatrix );

    return error;
}   // main()

/***********************************************************************************************/

/*
This function provides reference implementation of SpMV
functionality:
y = alpha * A * x + beta * y.
alpha, beta - scalars,
x - input vector,
y - resulting vector,
A - sparse matrix in 0-based CSR representation:
rows - number of rows in the matrix A
csrRows - integer array of length rows+1
csrCols - integer array of length nnz
csrVals - double  array of length nnz
*/

static void referenceSpMV ( const struct SparseMatrix *csrMatrix,
                            double       alpha,
                            const double *x,
                            double       beta,
                            double       *y )
{
    int i;
    int rows = csrMatrix->num_rows;
#pragma omp parallel for
    for ( i = 0; i < rows; i++ )
    {
        double yi = 0.0;
        int start = csrMatrix->rows[i];
        int end   = csrMatrix->rows[i + 1];
        int j;
        for ( j = start; j < end; j++ )
            yi += csrMatrix->vals[j] * x[csrMatrix->cols[j]];
        y[i] = yi * alpha + beta * y[i];
    }
}   // referenceSpMV

// Prints measured performance results for all SpMV benchmarks: Intel MKL, CSR, and ESB.
// parameter schedule = -1 indicates that no schedule-related message will be printed 
static void printPerformanceResults( const char* testName, double mflop, double bench_time, int niters, int schedule, char* filename, int numThreads )
{
    double time = bench_time / niters;      //  time for a single SpMV call
    double gflops = mflop / time / 1000;
    if ( bench_time < 0.5 )
    {
        printf( "Warning: measured time is less than 0.5 sec: %f\n", time );
        printf( "Measured results could be unstable\n" );
    }

//    printf( "%s performance results:\n", testName );
    printf( "The SpMV Execution time of %s  is %f seconds.   [file: %s] [threads: %d] ", testName, time, filename, numThreads);
    switch ( schedule )
    {
        case INTEL_SPARSE_SCHEDULE_STATIC:  printf( " [schedule: static]\n" );    break;
        case INTEL_SPARSE_SCHEDULE_DYNAMIC: printf( " [schedule: dynamic]\n" );   break;
        case INTEL_SPARSE_SCHEDULE_BLOCK:   printf( " [schedule: block]\n" );     break;
        default: printf("\n"); break;
    }
    printf( "         The Throughput of %s  is %f GFlops.    [file: %s] [threads: %d] ", testName, gflops, filename, numThreads );
    switch ( schedule )
    {
        case INTEL_SPARSE_SCHEDULE_STATIC:  printf( " [schedule: static]\n" );    break;
        case INTEL_SPARSE_SCHEDULE_DYNAMIC: printf( " [schedule: dynamic]\n" );   break;
        case INTEL_SPARSE_SCHEDULE_BLOCK:   printf( " [schedule: block]\n" );     break;
        default: printf("\n"); break;
    }
}   // printPerformanceResults

// Initialize input vectors
static void initVectors( int    rows,
                         int    cols,
                         double *x,
                         double *y,
                         double *y_ref )
{
    int i;
    for ( i = 0; i < rows; i++ )
    {
//        y[i] = M_PI;
//        y_ref[i] = M_PI;
        y[i] = 0;
        y_ref[i] = 0;
    }
    for ( i = 0; i < cols; i++ )
    {
//        x[i] = M_PI;
        x[i] = 1;
    }
}   // initVectors

// Calculate Frobenius norm of the matrix and vectors: 
static double calcFrobeniusNorm ( int vectorLength, double *vectorValues )
{
    int i;
    double norm = 0.0;
    for ( i = 0; i < vectorLength; i++ )
    {
        norm += vectorValues[i] * vectorValues[i];
    }
    return sqrt (norm) ;
}   // calcFrobeniusNorm

// Calculate Frobenius norm of vectors y and y_ref residual
static double calculateResidual ( int          size,
                                  const double *y,
                                  const double *y_ref )
{
    double res = 0.0;
    int i;

    for ( i = 0; i < size; i++ )
    {
        res += ( y[i] - y_ref[i] ) * ( y[i] - y_ref[i] );
    }

    res = sqrt ( res );

    return res;
}   // calculateResidual


// Validate and measure performance of Intel MKL SpMV implementation  
static int benchmark_MKL_SpMV ( const struct SparseMatrix *csrMatrix,
                                double       alpha,
                                double       *x,
                                double       beta,
                                double       *y,
                                double       *y_ref,
                                double       mflop,
                                double       matrixFrobeniusNorm,
                                char *       filename,
                                int          numIterations,
                                int          numThreads)
{
    char transa = 'n';  // Non-transpose Intel MKL SpMV functionality
    char matdescra[6] = { 'G', 'x', 'x', 'C', 'x', 'x'};    // General matrix, with 0-based indexing
    int  niters, iter;  // 
    double time_start, time_end, time, residual;
    double normX, normY;
    double estimatedAccuracy;

    // y = y_ref
    initVectors( csrMatrix->num_rows, csrMatrix->num_cols, x, y, y_ref );
    normX = calcFrobeniusNorm ( csrMatrix->num_cols, x );
    normY = calcFrobeniusNorm ( csrMatrix->num_rows, y );
    // estimate accuracy of SpMV operation: y = alpha * A * x + beta * y
    // | y1[i] - y2[i] | < eps * ( |alpha| * ||A|| * ||x|| + |beta| * ||y||)
    estimatedAccuracy = fabs( alpha ) * matrixFrobeniusNorm * normX + fabs( beta ) * normY;
    estimatedAccuracy *= EPS;
    // y_ref = alpha * A * x + beta * y_ref
    referenceSpMV ( csrMatrix, alpha, x, beta, y_ref );

    // y = alpha * A * x + beta * y
    mkl_dcsrmv( &transa, (int*)&csrMatrix->num_rows, (int*)&csrMatrix->num_cols, &alpha, matdescra,
                csrMatrix->vals, csrMatrix->cols, csrMatrix->rows, &csrMatrix->rows[1], x, &beta, y );
    // check for equality of y_ref and y
    residual = calculateResidual ( csrMatrix->num_rows, y, y_ref );
    if ( residual > estimatedAccuracy )
    {
        // The library implementation of SpMV probably computed a wrong result
        fprintf( stderr, "ERROR: the difference is too high. Residual %e is above threshold %f\n",
                 residual, estimatedAccuracy );
        return -1;
    }
    else
    {
        printf( "Validation PASSED\n" );
    }
    // estimate number of iterations: measured time should be long enough for stable measurement
//    niters = (int) ( 2.0e3 / mflop );

    printf(" ==============================================\n");

#ifdef NITERS
    niters = NITERS;
#else   
//    niters = 1000;
    niters = numIterations;
#endif

    printf(" iterations :  %d\n", niters);

    if ( niters < 1 )
        niters = 1;
    time_start = dsecnd();
    for ( iter = 0; iter < niters; iter++ )
        mkl_dcsrmv( &transa, (int*)&csrMatrix->num_rows, (int*)&csrMatrix->num_cols, &alpha, matdescra,
                    csrMatrix->vals, csrMatrix->cols, csrMatrix->rows, &csrMatrix->rows[1], x, &beta, y );
    time_end = dsecnd();
    time = time_end - time_start;
    // print performance results in GFlops and time per single SpMV call
    printPerformanceResults( "MKL  ", mflop, time, niters, -1, filename, numThreads);

//    printf("MKL EXE Time %s %f \n", filename, time/niters);

/*
    int kkkk = 0;
    for(kkkk=0; kkkk<256; kkkk+=16)
       printf(" y_ref[%d] = %f\n", kkkk, y_ref[kkkk]);
*/
    return 0;
}   // benchmark_MKL_SpMV

#ifdef __KNC__
// Validate and measure performance of experimental CSR SpMV implementation  
static int benchmark_CSR_SpMV ( const struct SparseMatrix *csrMatrix,
                                double       alpha,
                                double       *x,
                                double       beta,
                                double       *y,
                                double       *y_ref,
                                double       mflop,
                                double       matrixFrobeniusNorm,
                                sparseSchedule_t schedule,
                                char *       filename,
                                int          numIterations,
                                int          numThreads )
{
    sparseCSRMatrix_t csrA;     // Structure with CSR matrix
    sparseMatDescr_t descrA;    // CSR matrix descriptor
    int niters, iter;
    double time_start, time_end, time;
    double normX, normY, residual;
    double estimatedAccuracy;

    // y = y_ref
    initVectors( csrMatrix->num_rows, csrMatrix->num_cols, x, y, y_ref );
    // y_ref = alpha * A * x + beta * y_ref
    normX = calcFrobeniusNorm ( csrMatrix->num_cols, x );
    normY = calcFrobeniusNorm ( csrMatrix->num_rows, y );
    // estimate accuracy of SpMV oparation: y = alpha * A * x + beta * y
    // | y1[i] - y2[i] | < eps * ( |alpha| * ||A|| * ||x|| + |beta| * ||y||)
    estimatedAccuracy = fabs( alpha ) * matrixFrobeniusNorm * normX + fabs( beta ) * normY;
    estimatedAccuracy *= EPS;
    referenceSpMV ( csrMatrix, alpha, x, beta, y_ref );

    printf(" ==============================================\n");
    /* Functions below could return an error in the following situations:
       INTEL_SPARSE_STATUS_ALLOC_FAILED - not enough memory to allocate working arrays;
       INTEL_SPARSE_STATUS_EXECUTION_FAILED - implemented algorithm reported wrong result. */

    double time_pre = dsecnd();
    if ( sparseCreateCSRMatrix ( &csrA, schedule ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after creation of CSR matrix\n" );
        return -1;
    }

    if ( sparseCreateMatDescr ( &descrA ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf ( stderr, "Error after creation of matrix descriptor\n" ); 
        sparseDestroyCSRMatrix ( csrA );
        return -2;
    }

    if ( sparseDcsr2csr ( csrMatrix->num_rows, csrMatrix->num_cols, descrA, csrMatrix->vals,
                          csrMatrix->rows, csrMatrix->cols, csrA ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after conversion to CSR matrix\n" );
        sparseDestroyCSRMatrix ( csrA );
        sparseDestroyMatDescr ( descrA );
        return -3;
    }

//    printf(" Pre-processing Time of Internal CSR is %f\n", dsecnd()-time_pre);
//    printf("CSR PRE Time %s %f \n", filename, dsecnd() - time_pre);
    printf("The Pre-processing(CSR->CSR_I) Time of CSR_I is %f seconds.   [file: %s] [threads: %d]", dsecnd() - time_pre, filename, numThreads);
    switch ( schedule )
    {
        case INTEL_SPARSE_SCHEDULE_STATIC:  printf( " [schedule: static]\n" );    break;
        case INTEL_SPARSE_SCHEDULE_DYNAMIC: printf( " [schedule: dynamic]\n" );   break;
        case INTEL_SPARSE_SCHEDULE_BLOCK:   printf( " [schedule: block]\n" );     break;
        default: printf("\n"); break;
    }
//    printf(" ==============================================\n");


    // y = alpha * A * x + beta * y
    if ( sparseDcsrmv ( INTEL_SPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrA, x, &beta, y ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after SpMV in CSR format\n" );
        sparseDestroyCSRMatrix ( csrA );
        sparseDestroyMatDescr ( descrA );
        return -4;
    }

    // check for equality of y_ref and y
    residual = calculateResidual( csrMatrix->num_rows, y, y_ref );
    if ( residual > estimatedAccuracy )
    {
        // The library implementation of SpMV probably computed a wrong result
        fprintf( stderr, "ERROR: the difference is too high. Residual %e is above threshold %f\n",
                 residual, estimatedAccuracy );
        sparseDestroyCSRMatrix ( csrA );
        sparseDestroyMatDescr ( descrA );
        return -5;
    }
    else
    {
        printf( "Validation PASSED\n" );
    }
    // estimate number of iterations: measured time should be long enough for stable measurement
//    niters = (int) ( 2.0e3 / mflop );
#ifdef NITERS
    niters = NITERS;
#else   
//    niters = 1000;
    niters = numIterations;
#endif

    printf(" iterations :  %d\n", niters);

    if ( niters < 1 )
        niters = 1;
    time_start = dsecnd();
    for ( iter = 0; iter < niters; iter++ )
        sparseDcsrmv ( INTEL_SPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrA, x, &beta, y );
    time_end = dsecnd();
    time = time_end - time_start;
    // print performance results in GFlops and time per single SpMV call
    printPerformanceResults( "CSR_I", mflop, time, niters, schedule, filename, numThreads );
//    printf("CSR EXE Time %s %f \n", filename, time/niters);

    sparseDestroyCSRMatrix ( csrA );
    sparseDestroyMatDescr ( descrA );

    return 0;
}   // benchmark_CSR_SpMV

// Validate and measure performance of experimental ESB SpMV implementation
static int benchmark_ESB_SpMV ( const struct SparseMatrix *csrMatrix,
                                double       alpha,
                                double       *x,
                                double       beta,
                                double       *y,
                                double       *y_ref,
                                double       mflop,
                                double       matrixFrobeniusNorm,
                                sparseSchedule_t schedule,
                                char *       filename,
                                int          numIterations,
                                int          numThreads)
{
    sparseESBMatrix_t esbA;     // Structure with ESB matrix
    sparseMatDescr_t descrA;    // ESB matrix descriptor
    int niters, iter;
    double time_start, time_end, time;
    double normX, normY, residual;
    double estimatedAccuracy;

    // y = y_ref
    initVectors( csrMatrix->num_rows, csrMatrix->num_cols, x, y, y_ref );
    // y_ref = alpha * A * x + beta * y_ref

    normX = calcFrobeniusNorm ( csrMatrix->num_cols, x );
    normY = calcFrobeniusNorm ( csrMatrix->num_rows, y );
    // estimate accuracy of SpMV oparation: y = alpha * A * x + beta * y
    // || y1 - y2 || < eps * ( |alpha| * ||A|| * ||x|| + |beta| * ||y||)
    estimatedAccuracy = fabs( alpha ) * matrixFrobeniusNorm * normX + fabs( beta ) * normY;
    estimatedAccuracy *= EPS;

    referenceSpMV ( csrMatrix, alpha, x, beta, y_ref );

    printf(" ==============================================\n");
    /* Functions below could return an error in the following situations:
   INTEL_SPARSE_STATUS_ALLOC_FAILED - not enough memory to allocate working arrays;
   INTEL_SPARSE_STATUS_EXECUTION_FAILED - implemented algorithm reported wrong result. */


    double time_pre = dsecnd();

    if ( sparseCreateESBMatrix ( &esbA, schedule ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after creation of ESB matrix\n" );
        return -1;
    }

    if ( sparseCreateMatDescr ( &descrA ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after creation of matrix descriptor\n" );
        sparseDestroyESBMatrix ( esbA );
        return -2;
    }

    if ( sparseDcsr2esb ( csrMatrix->num_rows, csrMatrix->num_cols, descrA, csrMatrix->vals,
                          csrMatrix->rows, csrMatrix->cols, esbA ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after conversion to ESB matrix\n" );
        sparseDestroyESBMatrix ( esbA );
        sparseDestroyMatDescr ( descrA );
        return -3;
    }

//    printf(" **********************************************\n");
//    printf(" Pre-processing Time of ESB is %f\n", dsecnd()-time_pre);
//    printf(" **********************************************\n");
     
//    printf("ESB PRE Time %s %f \n", filename, dsecnd() - time_pre);
    printf("The Pre-processing(CSR->ESB)   Time of ESB   is %f seconds.   [file: %s] [threads: %d]", dsecnd() - time_pre, filename, numThreads);

    switch ( schedule )
    {
        case INTEL_SPARSE_SCHEDULE_STATIC:  printf( " [schedule: static]\n" );    break;
        case INTEL_SPARSE_SCHEDULE_DYNAMIC: printf( " [schedule: dynamic]\n" );   break;
        case INTEL_SPARSE_SCHEDULE_BLOCK:   printf( " [schedule: block]\n" );     break;
        default: printf("\n"); break;
    }

    // y = alpha * A * x + beta * y
    if ( sparseDesbmv ( INTEL_SPARSE_OPERATION_NON_TRANSPOSE, &alpha, esbA, x, &beta, y ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after SpMV in ESB format\n" );
        sparseDestroyESBMatrix ( esbA );
        sparseDestroyMatDescr ( descrA );
        return -4;
    }

    // check for equality of y_ref and y
    residual = calculateResidual ( csrMatrix->num_rows, y, y_ref );
    if ( residual > estimatedAccuracy )
    {
        // The library implementation of SpMV probably computed a wrong result
        fprintf( stderr, "ERROR: the difference is too high. Residual %e is above threshold %f\n",
                 residual, estimatedAccuracy );
        sparseDestroyESBMatrix ( esbA );
        sparseDestroyMatDescr ( descrA );
        return -5;
    }
    else
    {
        printf( "Validation PASSED\n" );
    }
    // estimate number of iterations: measured time should be long enough for stable measurement
//    niters = (int) ( 2.0e3 / mflop );
#ifdef NITERS
    niters = NITERS;
#else   
//    niters = 1000;
    niters = numIterations;
#endif

    printf(" iterations :  %d\n", niters);

    if ( niters < 1 )
        niters = 1;
    time_start = dsecnd();
    for ( iter = 0; iter < niters; iter++ )
        sparseDesbmv ( INTEL_SPARSE_OPERATION_NON_TRANSPOSE, &alpha, esbA, x, &beta, y );
    time_end = dsecnd();
    time = time_end - time_start;
    // print performance results in GFlops and time per single SpMV call
    printPerformanceResults( "ESB  ", mflop, time, niters, schedule, filename, numThreads );
//    printf("ESB EXE Time %s %f \n", filename, time/niters);

    sparseDestroyESBMatrix ( esbA );
    sparseDestroyMatDescr ( descrA );
    return 0;
}   // benchmark_ESB_SpMV
#endif

/*
Sparse Matrix auxiliary routines:

    readSparseCOOMatrix - read input matrix in COO format from the file;
        Input file should be in Matrix Matrix format.

    convertCOO2CSR - create matrix in CSR format from COO representation.
        Elements in rows of CSR matrix are sorted by column numbers in
        increasing order

    deleteSparseMatrix - free memory previously allocated for the matrix

    printMatrixInfo - prints short sparse matrix statistics
*/

static int readSparseCOOMatrix ( FILE *f, struct SparseMatrix *cooMatrix )
{
    MM_typecode matcode;
    int isComplex, isInteger, isReal, isSymmetric, isPattern;
    int sizeM, sizeN, sizeV, nnz, ret_code, counter, idum, i;
    double ddum;
    int *rows = NULL, *cols = NULL;
    double *vals = NULL;

    if ( mm_read_banner(f, &matcode) != 0 )
    {
        fprintf( stderr, "Could not process matrix market banner.\n");
        return -1;
    }
    if ( !mm_is_matrix( matcode ) )
    {
        fprintf( stderr, "Could not process non-matrix input.\n");
        return -2;
    }

    if ( !mm_is_sparse( matcode ) )
    {
        fprintf( stderr, "Could not process non-sparse matrix input.\n");
        return -3;
    }

    isComplex = 0;
    isReal    = 0;
    isInteger = 0;
    isSymmetric = 0;
    isPattern = 0;

    if ( mm_is_complex( matcode ) )
    {
        isComplex = 1;
    }

    if ( mm_is_real ( matcode) )
    {
        isReal = 1;
    }

    if ( mm_is_integer ( matcode ) )
    {
        isInteger = 1;
    }

    if ( mm_is_pattern ( matcode ) )
    {
        isPattern = 1;
    }

    /* find out size of sparse matrix .... */

    if ( ( ret_code = mm_read_mtx_crd_size( f, &sizeM, &sizeN, &sizeV ) ) != 0 )
    {
        fprintf( stderr, "Could not process matrix sizes\n");
        return -4;
    }

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        nnz = sizeV * 2; // up to two times more elements than in half of the matrix
    }
    else
    {
        nnz = sizeV;
    }

    /* allocate memory for matrices */

    rows = ( int* )    MKL_malloc( nnz * sizeof( int ), ALIGN );
    cols = ( int* )    MKL_malloc( nnz * sizeof( int ), ALIGN );
    vals = ( double* ) MKL_malloc( nnz * sizeof( double ), ALIGN );

    if ( NULL == rows || NULL == cols || NULL == vals )
    {
        MKL_free( rows );
        MKL_free( cols );
        MKL_free( vals );
        fprintf( stderr, "Could not allocate memory for input matrix arrays in COO format\n" );
        fprintf( stderr, "Rows = %d, Columns = %d, NNZ = %d\n", sizeM, sizeN, nnz );
        return -5;
    }

    counter = 0;

    for ( i = 0; i < sizeV; i++ )
    {
        if ( isComplex )
        {
            fscanf(f, "%d %d %lg %lg\n", &rows[counter], &cols[counter], &vals[counter], &ddum );
        }
        else if ( isReal )
        {
            fscanf(f, "%d %d %lg\n", &rows[counter], &cols[counter], &vals[counter] );
        }
        else if ( isInteger )
        {
            fscanf(f, "%d %d %d\n", &rows[counter], &cols[counter], &idum );
            vals[counter] = idum;
        }
        else if ( isPattern )
        {
            fscanf(f, "%d %d\n", &rows[counter], &cols[counter] );
            vals[counter] = 1;
        }
        counter++;
        if ( isSymmetric && rows[counter-1] != cols[counter-1] )
        // expand symmetric formats to "general" one
        {
            rows[counter] = cols[counter-1];
            cols[counter] = rows[counter-1];
            vals[counter] = vals[counter-1];
            counter++;
        }
    }

    if ( f !=stdin ) fclose(f);

    printf("Reading matrix completed\n" );

    cooMatrix->num_rows = sizeM;
    cooMatrix->num_cols = sizeN;
    cooMatrix->nnz   = counter; // Actual number of non-zeroes elements in COO matrix
    cooMatrix->rows  = rows;
    cooMatrix->cols  = cols;
    cooMatrix->vals  = vals;
    return 0;
}   // readSparseCOOMatrix

static int convertCOO2CSR ( const struct SparseMatrix *cooMatrix, struct SparseMatrix *csrMatrix )
{
    int info;
    int job[8];

    /************************/
    /* now convert matrix in COO 1-based format to CSR 0-based format */
    /************************/

    job[0] = 2; // COO -> sorted CSR
    job[1] = 0; // 0-based CSR
    job[2] = 0; // 1-based COO
    job[4] = cooMatrix->nnz;
    job[5] = 0; // all CSR arrays are filled

    info = 0;

    csrMatrix->num_rows = cooMatrix->num_rows;
    csrMatrix->num_cols = cooMatrix->num_cols;
    csrMatrix->nnz   = cooMatrix->nnz;

    csrMatrix->rows = ( int* )    MKL_malloc( ( cooMatrix->num_rows + 1 ) * sizeof( int ), ALIGN );
    csrMatrix->cols = ( int* )    MKL_malloc( cooMatrix->nnz * sizeof( int ),           ALIGN );
    csrMatrix->vals = ( double* ) MKL_malloc( cooMatrix->nnz * sizeof( double ),        ALIGN );

    if ( NULL == csrMatrix->rows || NULL == csrMatrix->cols || NULL == csrMatrix->vals )
    {
        MKL_free( csrMatrix->rows );
        MKL_free( csrMatrix->cols );
        MKL_free( csrMatrix->vals );
        fprintf( stderr, "Could not allocate memory for converting matrix to CSR format\n" );
        return -5;
    }

    mkl_dcsrcoo ( job,
                  &csrMatrix->num_rows,
                  csrMatrix->vals,
                  csrMatrix->cols,
                  csrMatrix->rows,
                  (int*)&cooMatrix->nnz,
                  cooMatrix->vals,
                  cooMatrix->rows,
                  cooMatrix->cols,
                  &info );

    if ( info != 0 )
    {
        fprintf( stderr, "Error converting COO -> CSR: %d\n", info );
        MKL_free( csrMatrix->rows );
        MKL_free( csrMatrix->cols );
        MKL_free( csrMatrix->vals );
        return -10;
    }

/*
    int kkkk=0;
    for(kkkk=0; kkkk<256; kkkk++)
       printf(" cols[%d]= %d, vals[%d]=%f\n", kkkk, csrMatrix->cols[kkkk], kkkk, csrMatrix->vals[kkkk]);
*/

    printf( "Operation COO->CSR completed\n" );
    return 0;
}   // convertCOO2CSR

static void deleteSparseMatrix ( struct SparseMatrix *matrix )
{
    MKL_free( matrix->rows );
    MKL_free( matrix->cols );
    MKL_free( matrix->vals );
}   // deleteSparseMatrix

static void printMatrixInfo ( const struct SparseMatrix *csrMatrix )
{
    int omp_threads;

    int env_threads = omp_get_max_threads();

    mkl_set_num_threads_local(env_threads);

    omp_threads = mkl_get_max_threads();

    printf( "Number of OMP threads: %d\n", omp_threads );
    printf( "Sparse matrix info:\n" );
    printf( "       rows: %d\n", csrMatrix->num_rows );
    printf( "       cols: %d\n", csrMatrix->num_cols );
    printf( "       nnz:  %d\n", csrMatrix->nnz );
}   // printMatrixInfo
