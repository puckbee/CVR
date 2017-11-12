
#pragma once

#include <xmmintrin.h>

#define MALLOC(SIZE) _mm_malloc((SIZE), 64)
#define FREE(PTR)    _mm_free((PTR))

//#define MALLOC(SIZE) malloc((SIZE))
//#define FREE(PTR)    free((PTR))
