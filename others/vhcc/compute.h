
typedef int index_t;
typedef double value_t;

#define LEN16 16
#define LEN8 8


typedef __attribute__((aligned(64))) union zmmi {
	__m512i reg;
	unsigned int elems[LEN16];
} zmmi_t;
typedef __attribute__((aligned(64))) union zmmd {
	__m512d reg;
	__m512i regi32;
	double elems[LEN8];
} zmmd_t;

int count_trailing_zero(int a, __mmask8 x)
{
   int idx = a+1;
   __mmask8 mask[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
   while((x & mask[idx]) == 0)
   {
      idx ++;
   }
  
   return idx;
   
}


void compute_spmv(int n_threads, int num_vectors,
									int threads_per_core,
									int num_panels,
									panel_info_t *restrict panel_info,
									thr_info_t   *restrict thr_info,
									index_t *restrict veceor_ptr,
									uint8_t *restrict scan_mask,
									index_t *restrict row_arr,
									index_t *restrict col_arr,
									value_t *restrict vals_arr,
									value_t *restrict input,
									value_t *restrict result)
{
#pragma omp parallel default(shared) num_threads(n_threads)
	{

		int id = omp_get_thread_num();

		int core_id = id / threads_per_core;
		int local_thr_id = id % threads_per_core;
		
		int panel_id = thr_info[id].panel_id;
		
		value_t *tmp_result = panel_info[panel_id].tmp_result;
		
		index_t start_vec = thr_info[id].start_vec;
		index_t end_vec   = thr_info[id].end_vec;
		
		zmmi_t row, col, wrmask;
		zmmd_t res, tmp;
		__mmask8 mask1, mask2, mask3, maskwr;
		
		index_t veceor_idx = thr_info[id].vbase;
		index_t scan_idx   = thr_info[id].sbase;
		index_t ridx       = thr_info[id].rbase;
		index_t vec_idx    = start_vec * LEN8;
		
		value_t nrval = 0;
		index_t eor_vec = veceor_ptr[veceor_idx++];
		res.elems[:] = 0;
		for (index_t v = start_vec; v < end_vec; ++v) {
			
			col.elems[0:LEN8] = col_arr[vec_idx:LEN8];
			
			__assume_aligned(&vals_arr[vec_idx], 64);
			
			res.elems[0:LEN8] += vals_arr[vec_idx:LEN8] * 
				input[col.elems[0:LEN8]];
			vec_idx += LEN8;
			
			nrval = 0;
			if (v == eor_vec) {
				mask1 = (__mmask8)scan_mask[scan_idx++];
				mask2 = (__mmask8)scan_mask[scan_idx++];
				mask3 = (__mmask8)scan_mask[scan_idx++];
				maskwr = (__mmask8)scan_mask[scan_idx++];
				
				res.reg = _mm512_mask_add_pd(res.reg, mask1, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
				res.reg = _mm512_mask_add_pd(res.reg, mask2, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
				tmp.regi32 = _mm512_permute4f128_epi32(res.regi32, _MM_PERM_BBBA);
				res.reg = _mm512_mask_add_pd(res.reg, mask3, res.reg, _mm512_swizzle_pd(tmp.reg, _MM_SWIZ_REG_BBBB));
				
				if ((maskwr & 0x80) == 0)
					nrval = res.elems[LEN8-1];

				int bcnt = _mm_countbits_32(maskwr);
//				int a = -1;
				int a = -1;
				int x = maskwr;
				for (int i = 0; i < bcnt; ++i) {
//					int y = _mm_tzcnti_32(a, x);
					int y = count_trailing_zero(a,maskwr);
					index_t r = row_arr[ridx+i];
					tmp_result[r] += res.elems[y];
					a = y;
				}
				ridx += bcnt;

				eor_vec = veceor_ptr[veceor_idx++];
				
			} else {

				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
				nrval = res.elems[LEN8-1] + res.elems[3];
			}
			
			res.elems[:] = 0;
			res.elems[0] = nrval;
		}
		
#pragma omp barrier

		index_t nridx = thr_info[id].last_row;
		nrval = tmp_result[thr_info[id].overflow_row];

#pragma omp atomic update
		tmp_result[nridx] += nrval;
	
#pragma omp barrier
		
		index_t merge_start = thr_info[id].merge_start;
		index_t merge_end   = thr_info[id].merge_end;
		index_t blk_size    = 512;
		
		for (index_t i = merge_start; i < merge_end; i += blk_size) {
			index_t blk_end = i + blk_size > merge_end ? merge_end : i + blk_size;
			for (int c = 0; c < num_panels; ++c) {
				for (index_t b = i; b < blk_end; b += LEN8) {
					result[b:LEN8] += panel_info[c].tmp_result[b:LEN8];
				}
			}
		}
	}
}




void compute_spmv1(int n_threads, int num_vectors,
									 thr_info_t *restrict thr_info,
									 index_t *restrict veceor_ptr,
									 uint8_t *restrict scan_mask,
									 index_t *restrict row_arr,
									 index_t *restrict col_arr,
									 value_t *restrict vals_arr,
									 value_t *restrict input,
									 value_t *restrict result)
{
#pragma omp parallel default(shared) num_threads(n_threads)
	{

		int id = omp_get_thread_num();

		index_t start_vec = thr_info[id].start_vec;
		index_t end_vec   = thr_info[id].end_vec;

		zmmi_t row, col, wrmask;
		zmmd_t res, tmp;
		__mmask8 mask1, mask2, mask3, maskwr;

		index_t cidx       = thr_info[id].vbase;
		index_t veceor_idx = thr_info[id].vbase;
		index_t scan_idx   = thr_info[id].vbase * 4;
		index_t ridx       = thr_info[id].rbase;
		index_t vec_idx    = start_vec * LEN8;

		value_t nrval = 0;
		index_t eor_vec = veceor_ptr[veceor_idx++];
		res.elems[:] = 0;
                 
//                std::cout<<" start = "<< start_vec <<";  end = "<< end_vec<<endl;

		for (index_t v = start_vec; v < end_vec; ++v) {
			
			col.elems[0:LEN8] = col_arr[vec_idx:LEN8];

			__assume_aligned(&vals_arr[vec_idx], 64);
			res.elems[0:LEN8] += vals_arr[vec_idx:LEN8] * input[col.elems[0:LEN8]];
			vec_idx += LEN8;

			nrval = 0;
			if (v == eor_vec) {
				mask1 = (__mmask8)scan_mask[scan_idx++];
				mask2 = (__mmask8)scan_mask[scan_idx++];
				mask3 = (__mmask8)scan_mask[scan_idx++];
				maskwr = (__mmask8)scan_mask[scan_idx++];
				
				res.reg = _mm512_mask_add_pd(res.reg, mask1, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
				res.reg = _mm512_mask_add_pd(res.reg, mask2, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
				tmp.regi32 = _mm512_permute4f128_epi32(res.regi32, _MM_PERM_BBBA);
				res.reg = _mm512_mask_add_pd(res.reg, mask3, res.reg, _mm512_swizzle_pd(tmp.reg, _MM_SWIZ_REG_BBBB));

				if ((maskwr & 0x80) == 0)
					nrval = res.elems[LEN8-1];

				int bcnt = _mm_countbits_32(maskwr);
//				int a = -1;
				int a = -1;
				int x = maskwr;
				for (int i = 0; i < bcnt; ++i) {
//					int y = _mm_tzcnti_32(a, x);
					int y = count_trailing_zero(a,maskwr);
//                                            std::cout<<"bcnt = "<< bcnt<<"; y = "<< y<<"; v= "<< v<<"; start = "<< start_vec<<"; end = "<< end_vec<<endl;      

					index_t r = row_arr[ridx+i];
					result[r] += res.elems[y];
					a = y;
				}
				ridx += bcnt;

				eor_vec = veceor_ptr[veceor_idx++];
				
			} else {
				
//				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
//				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
//				nrval = res.elems[LEN8-1] + res.elems[3];
                                nrval = _mm512_reduce_add_pd(res.reg);				
			}

			res.elems[:] = 0;
			res.elems[0] = nrval;
		}

#pragma omp barrier

		index_t nridx = thr_info[id].last_row;
		nrval = result[thr_info[id].overflow_row];
#pragma omp atomic update
		result[nridx] += nrval;
	}
}


void run_spmv_vhcc1(int n_threads, int num_vectors,
									 thr_info_t *restrict thr_info,
									 index_t *restrict veceor_ptr,
									 uint8_t *restrict scan_mask,
									 index_t *restrict row_arr,
									 index_t *restrict col_arr,
									 value_t *restrict vals_arr,
									 value_t *restrict input,
									 value_t *restrict result,
                                                                         int iters)
{
     for (int i = 0; i < iters; ++i) {
          compute_spmv1(n_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
     }



}
void run_spmv_vhcc(int n_threads, int num_vectors,
									int threads_per_core,
									int num_panels,
									panel_info_t *restrict panel_info,
									thr_info_t   *restrict thr_info,
									index_t *restrict veceor_ptr,
									uint8_t *restrict scan_mask,
									index_t *restrict row_arr,
									index_t *restrict col_arr,
									value_t *restrict vals_arr,
									value_t *restrict input,
									value_t *restrict result,
                                                                        int iters)
{


		for (int i = 0; i < iters; ++i) {
			compute_spmv(n_threads, num_vectors, threads_per_core, num_panels, panel_info, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
		}

}
