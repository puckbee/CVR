/********************************************************************************
!   Copyright(C) 2013 Intel Corporation. All Rights Reserved.
!   
!   The source code, information  and  material ("Material") contained herein is
!   owned  by Intel Corporation or its suppliers or licensors, and title to such
!   Material remains  with Intel Corporation  or its suppliers or licensors. The
!   Material  contains proprietary information  of  Intel or  its  suppliers and
!   licensors. The  Material is protected by worldwide copyright laws and treaty
!   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
!   modified, published, uploaded, posted, transmitted, distributed or disclosed
!   in any way  without Intel's  prior  express written  permission. No  license
!   under  any patent, copyright  or  other intellectual property rights  in the
!   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
!   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
!   intellectual  property  rights must  be express  and  approved  by  Intel in
!   writing.
!   
!   *Third Party trademarks are the property of their respective owners.
!   
!   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
!   this  notice or  any other notice embedded  in Materials by Intel or Intel's
!   suppliers or licensors in any way.
!
!*******************************************************************************
!
!   This file contains interfaces to experimental high-optimized sparse
!   matrix-vector multiplication (SpMV) kernels for CSR and ESB formats,
!   general non-transposed cases.
! 
*******************************************************************************/ 
#ifndef _SPMV_INTERFACE_H
#define _SPMV_INTERFACE_H


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

    typedef enum
    {
        INTEL_SPARSE_STATUS_SUCCESS=0,
        INTEL_SPARSE_STATUS_NOT_INITIALIZED=1,
        INTEL_SPARSE_STATUS_ALLOC_FAILED=2,
        INTEL_SPARSE_STATUS_INVALID_VALUE=3,
        INTEL_SPARSE_STATUS_ARCH_MISMATCH=4,  // Not implemented currently
        INTEL_SPARSE_STATUS_MAPPING_ERROR=5,  // Not implemented currently 
        INTEL_SPARSE_STATUS_EXECUTION_FAILED=6,
        INTEL_SPARSE_STATUS_INTERNAL_ERROR=7,  
        INTEL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8 // Not implemented currently 
    } sparseStatus_t;

    typedef enum
    {
        INTEL_SPARSE_OPERATION_NON_TRANSPOSE = 0,  
        INTEL_SPARSE_OPERATION_TRANSPOSE = 1,             // Not supported
        INTEL_SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2    // Not supported
    } sparseOperation_t;

    typedef enum
    {
        INTEL_SPARSE_SCHEDULE_STATIC   = 0,  
        INTEL_SPARSE_SCHEDULE_DYNAMIC  = 1,             
        INTEL_SPARSE_SCHEDULE_BLOCK    = 2        // Not supported for ESB
    } sparseSchedule_t;

    typedef enum
    {
        INTEL_SPARSE_MATRIX_TYPE_GENERAL = 0, 
        INTEL_SPARSE_MATRIX_TYPE_SYMMETRIC = 1,           // Not supported
        INTEL_SPARSE_MATRIX_TYPE_HERMITIAN = 2,           // Not supported
        INTEL_SPARSE_MATRIX_TYPE_TRIANGULAR = 3           // Not supported
    } sparseMatrixType_t;


    typedef enum
    {
        INTEL_SPARSE_INDEX_BASE_ZERO = 0, 
        INTEL_SPARSE_INDEX_BASE_ONE = 1
    } sparseIndexBase_t;

/* Opaque structure for sparse matrix in ESB format */
    struct sparseESBMatrix;
    typedef struct sparseESBMatrix *sparseESBMatrix_t;

/* Opaque structure for sparse matrix in CSR format */
    struct sparseCSRMatrix;
    typedef struct sparseCSRMatrix *sparseCSRMatrix_t;

/* Opaque structure holding the matrix descriptor */
    struct sparseMatDescr;
    typedef struct sparseMatDescr *sparseMatDescr_t;

/* ESB format */
    sparseStatus_t sparseCreateESBMatrix (sparseESBMatrix_t *esbA, sparseSchedule_t schedule);
    sparseStatus_t sparseDestroyESBMatrix(sparseESBMatrix_t  esbA);

    sparseESBMatrix_t sparsecreateesbmatrix_ (sparseSchedule_t *schedule);
    sparseStatus_t sparsedestroyesbmatrix_ (sparseESBMatrix_t  *esbA);

/* CSR format */
    sparseStatus_t sparseCreateCSRMatrix (sparseCSRMatrix_t *csrA, sparseSchedule_t schedule);
    sparseStatus_t sparseDestroyCSRMatrix(sparseCSRMatrix_t  csrA);

    sparseCSRMatrix_t sparsecreatecsrmatrix_ (sparseSchedule_t *schedule);
    sparseStatus_t sparsedestroycsrmatrix_ (sparseCSRMatrix_t  *csrA);

/* When the matrix descriptor is created, its fields are initialized to: 
   INTEL_SPARSE_MATRIX_TYPE_GENERAL
   INTEL_SPARSE_INDEX_BASE_ZERO
   All other fields are uninitialized
*/                                   
    sparseStatus_t sparseCreateMatDescr  (sparseMatDescr_t *descrA);
    sparseStatus_t sparseDestroyMatDescr (sparseMatDescr_t  descrA);

    sparseMatDescr_t sparsecreatematdescr_ ( );
    sparseStatus_t sparsedestroymatdescr_ ( sparseMatDescr_t  *descrA );

    sparseStatus_t      sparseSetMatType(sparseMatDescr_t descrA, sparseMatrixType_t type);
    sparseMatrixType_t  sparseGetMatType(const sparseMatDescr_t descrA);

    sparseStatus_t      sparsesetmattype_(sparseMatDescr_t *descrA, sparseMatrixType_t *type);
    sparseMatrixType_t  sparsegetmattype_(const sparseMatDescr_t *descrA);

    sparseStatus_t      sparseSetMatIndexBase(sparseMatDescr_t descrA, sparseIndexBase_t base);
    sparseIndexBase_t   sparseGetMatIndexBase(const sparseMatDescr_t descrA);

    sparseStatus_t      sparsesetmatindexbase_(sparseMatDescr_t *descrA, sparseIndexBase_t *base);
    sparseIndexBase_t   sparsegetmatindexbase_(const sparseMatDescr_t *descrA);


// Create sparse matrix into internal ESB format from 
    sparseStatus_t sparseDcsr2esb(   int m,
                                     int n,
                                     const sparseMatDescr_t descrA,
                                     const double *csrValA,
                                     const int *csrRowPtrA,
                                     const int *csrColIndA,
                                     sparseESBMatrix_t  esbA );

    sparseStatus_t sparsedcsr2esb_(  int *m,
                                     int *n,
                                     const sparseMatDescr_t *descrA,
                                     const double *csrValA,
                                     const int *csrRowPtrA,
                                     const int *csrColIndA,
                                     sparseESBMatrix_t  *esbA );

// Create sparse matrix into internal CSR format 
    sparseStatus_t sparseDcsr2csr(   int m,
                                     int n,
                                     const sparseMatDescr_t descrA,
                                     const double *csrValA,
                                     const int *csrRowPtrA,
                                     const int *csrColIndA,
                                     sparseCSRMatrix_t  csrA );

    sparseStatus_t sparsedcsr2csr_(  int *m, int *n, const sparseMatDescr_t *descrA, 
                                     const double *csrValA, const int *csrRowPtrA,
                                     const int *csrColIndA, sparseCSRMatrix_t  *csrA );

// Sparse matrix-vector multiplication

    sparseStatus_t sparseDesbmv( sparseOperation_t transA,
                                 const double *alpha,
                                 const sparseESBMatrix_t  esbA,
                                 const double *x,
                                 const double *beta,
                                 double *y);

    sparseStatus_t sparsedesbmv_( sparseOperation_t *transA,
                                  const double *alpha,
                                  const sparseESBMatrix_t  *esbA,
                                  const double *x,
                                  const double *beta,
                                  double *y);

    sparseStatus_t sparseDcsrmv( sparseOperation_t transA,
                                 const double *alpha,
                                 const sparseCSRMatrix_t  csrA,
                                 const double *x,
                                 const double *beta,
                                 double *y);

    sparseStatus_t sparsedcsrmv_( const sparseOperation_t *transA,
                                  const double *alpha,
                                  const sparseCSRMatrix_t  *csrA,
                                  const double *x,
                                  const double *beta,
                                  double *y);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
