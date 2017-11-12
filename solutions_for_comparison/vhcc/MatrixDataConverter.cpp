
#include "MatrixDataConverter.h"
#include "mmio.h"
#include "util.h"
#include "mem.h"


MatrixDataConverter::MatrixDataConverter() {
}

MatrixDataConverter::~MatrixDataConverter() {
}

uint64_t MatrixDataConverter::EncodeFromFile(unsigned char** out_buf, const char* fname) {
    int m, n, nnz;
    int *row_idx, *col_idx;
    double *tvals;

    if (mm_read_sparse_matrix(fname, &m, &n, &nnz, &row_idx, &col_idx, &tvals) != 0) {
        THROW_RUNTIME_ERROR("mm_read_sparse_matrix fails");
    }
    
    const uint64_t mem_size = sizeof(uint64_t) + 3*sizeof(int) + 2*sizeof(int)*nnz + sizeof(double)*nnz;
    unsigned char* buf = (unsigned char*)MALLOC(mem_size);
    unsigned char* ptr = buf;
        
    // Put everything to the buffer
    ((uint64_t*)ptr)[0] = mem_size;
    ptr += sizeof(uint64_t);
    ((int*)ptr)[0] = m;
    ((int*)ptr)[1] = n;
    ((int*)ptr)[2] = nnz;
    ptr += (3*sizeof(int));
    memcpy(ptr, row_idx, sizeof(int)*nnz);
    ptr += (sizeof(int)*nnz);
    memcpy(ptr, col_idx, sizeof(int)*nnz);
    ptr += (sizeof(int)*nnz);
    memcpy(ptr, tvals, sizeof(double)*nnz);
    ptr += (sizeof(double)*nnz);
    assert(ptr == (buf + mem_size));
    
    *out_buf = buf;
    return mem_size;
}

void MatrixDataConverter::DecodeFromFile(const char* fileName, int* M, int* N, int* nz, int** I, int** J, double** val) {
    // Open and get the file size
    FILE* file = NULL;
    file = fopen(fileName, "rb");
    if(!file) {
        THROW_RUNTIME_ERROR("opening file fails");
    }
    fseek(file, 0, SEEK_END);
    const uint64_t mem_size = ftell(file);
    rewind(file);
    
    // Read the whole file
    unsigned char* buf = (unsigned char*)malloc(mem_size);
    if(fread(buf, 1, mem_size, file) != mem_size) {
        THROW_RUNTIME_ERROR("reading file fails");
    }
    fclose(file);
    
    // Decode to the output
    // @note we reallocate I, J and val, to ensure the memory alignment
    unsigned char* ptr = buf;    
    if(((uint64_t*)ptr)[0] != mem_size) {
        THROW_RUNTIME_ERROR("data size is wrong, something terrible occured");
    }
    ptr += sizeof(uint64_t);
    *M = ((int*)ptr)[0];
    *N = ((int*)ptr)[1];
    *nz = ((int*)ptr)[2];
    ptr += 3*sizeof(int);
    const int nz_ = *nz;
    *I = (int*)MALLOC(sizeof(int)*nz_);
    *J = (int*)MALLOC(sizeof(int)*nz_);
    *val = (double*)MALLOC(sizeof(double)*nz_);
    memcpy(*I, ptr, sizeof(int)*nz_);
    ptr += sizeof(int)*nz_;
    memcpy(*J, ptr, sizeof(int)*nz_);
    ptr += sizeof(int)*nz_;
    memcpy(*val, ptr, sizeof(double)*nz_);
    ptr += sizeof(double)*nz_;
    assert(ptr == buf + mem_size);
    
    // Cleanup
    free(buf);
}
