
#ifndef SPARSEMATRIXREADER_H
#define	SPARSEMATRIXREADER_H

#include "util.h"

class SparseMatrixReader {
public:
    SparseMatrixReader();
    virtual ~SparseMatrixReader();
    
    static bool ReadRawData(const char *fname, int *M, int *N, int *nz, int **I, int **J, double **val);
    
    static bool ReadEncodedData(const char *fname, int *M, int *N, int *nz, int **I, int **J, double **val);
    
private:
    DISALLOW_COPY_AND_ASSIGN(SparseMatrixReader);
};

#endif	/* SPARSEMATRIXREADER_H */

