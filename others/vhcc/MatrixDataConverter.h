
#ifndef MATRIXDATACONVERTER_H
#define	MATRIXDATACONVERTER_H

#include "util.h"
#include <cstdint>

class MatrixDataConverter {
public:
    MatrixDataConverter();
    virtual ~MatrixDataConverter();
    

    /**
     * 
     * @param buf
     * @param fileName
     * @return The size of buf
     */
    static uint64_t EncodeFromFile(unsigned char** out_buf, const char* fname);
    
    static void DecodeFromFile(const char *fileName, int *M, int *N, int *nz, int **I, int **J, double **val);
    
private:
    DISALLOW_COPY_AND_ASSIGN(MatrixDataConverter);
};

#endif	/* MATRIXDATACONVERTER_H */

