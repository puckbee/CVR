
#include "SparseMatrixReader.h"
#include "mmio.h"
#include "MatrixDataConverter.h"

SparseMatrixReader::SparseMatrixReader() {
}

SparseMatrixReader::~SparseMatrixReader() {
}

bool SparseMatrixReader::ReadRawData(const char* fname, int* M, int* N, int* nz, int** I, int** J, double** val) {
    if (mm_read_sparse_matrix(fname, M, N, nz, I, J, val) != 0) {
        return false;
    } else {
        return true;
    }
}


bool SparseMatrixReader::ReadEncodedData(const char* fname, int* M, int* N, int* nz, int** I, int** J, double** val) {
    MatrixDataConverter::DecodeFromFile(fname, M, N, nz, I, J, val);
    return true;
}

