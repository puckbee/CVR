#pragma once

#include "util.h"
#include "mem.h"
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <map>

#include <time.h>
#include <sys/time.h>
#define MICRO_IN_SEC 1000000.00

double microtime(){
        int tv_sec,tv_usec;
        double time;
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}


using namespace std;

const int MAX_CORES = 60;
const int MAX_THREADS_PER_CORE = 4;
//const int MAX_THREADS_PER_CORE = 1;


typedef struct {
	int vbase;
	int rbase;
	int sbase;
	int last_row;
	int overflow_row;
	int start_vec;
	int end_vec;
	int panel_id;
	int merge_start;
	int merge_end;
} thr_info_t;

typedef struct {
	double *tmp_result;
} panel_info_t;

template<typename IndexType = int, typename ValueType = double>
	class vhcc_matrix {
public:

const static int VECLEN = 512/8/sizeof(ValueType);
typedef IndexType index_t;
typedef ValueType value_t;
typedef uint8_t mask_t;

vhcc_matrix(index_t num_rows, index_t num_cols, index_t num_entries,
					 index_t *row_idx, index_t *col_idx, value_t *vals) 
: _num_rows(num_rows), _num_cols(num_cols), _num_entries(num_entries),
_row_idx(row_idx), _col_idx(col_idx), _vals(vals) {

	_thr_info = NULL;
	_veceor_ptr = NULL;
	_row_arr = NULL;
	_col_arr = NULL;
	_vals_arr = NULL;
}

~vhcc_matrix() {
	FREE(_row_idx); 
	FREE(_col_idx);
	FREE(_vals);

	if (_thr_info != NULL)
		FREE(_thr_info);
	if (_veceor_ptr != NULL)
		FREE(_veceor_ptr);
	if (_scan_mask != NULL)
		FREE(_scan_mask);
	if (_row_arr != NULL)
		FREE(_row_arr);
	if (_col_arr != NULL)
		FREE(_col_arr);
	if (_vals_arr != NULL)
		FREE(_vals_arr);
}

index_t get_num_rows() { return _num_rows; }
index_t get_num_cols() { return _num_cols; }
index_t get_num_entries() { return _num_entries; }
index_t *get_row_idx() { return _row_idx; }
index_t *get_col_idx() { return _col_idx; }
value_t *get_vals() { return _vals; }

int get_num_panels() { return _num_panels; }
panel_info_t *get_panel_info() { return _panel_info; }
int get_num_threads() { return _num_threads; }
int get_num_vectors() { return _num_vectors; }
index_t get_pad_entries() { return _pad_entries; }
index_t get_pad_rows() { return _pad_rows; }
index_t get_pad_cols() { return _pad_cols; }
thr_info_t *get_thr_info() { return _thr_info; }
index_t     get_thr_info_size() { return _thr_info_size; }
index_t *get_veceor_ptr() { return _veceor_ptr; }
index_t  get_veceor_size() { return _veceor_size; }
uint8_t *get_scan_mask() { return _scan_mask; }
index_t  get_scan_mask_size() { return _scan_mask_size; }
index_t *get_row_arr() { return _row_arr; }
index_t  get_row_arr_size() { return _row_arr_size; }
index_t *get_col_arr() { return _col_arr; }
index_t  get_col_arr_size() { return _col_arr_size; }
value_t *get_vals_arr() { return _vals_arr; }
index_t  get_vals_arr_size() { return _vals_arr_size; }

void convert(int num_threads, int threads_per_core, int num_panel, int rlength, int clength, char* filename);

private:

index_t _num_rows;
index_t _num_cols;
index_t _num_entries;
index_t *_row_idx;
index_t *_col_idx;
value_t *_vals;

int      _rlength;
int      _clength;
int      _num_cores;
int      _num_panels;
panel_info_t *_panel_info;
int      _num_threads;
index_t  _num_vectors;
index_t  _nnz_per_panel;
index_t  _pad_entries;
index_t  _extended_rows;
index_t  _pad_rows;
index_t  _pad_cols;
thr_info_t *_thr_info;
index_t  _thr_info_size;
index_t *_veceor_ptr;
index_t  _veceor_size;
uint8_t *_scan_mask;
index_t  _scan_mask_size;
index_t *_tmp_row_arr;
index_t *_row_arr;
index_t  _row_arr_size;
index_t *_col_arr;
index_t  _col_arr_size;
value_t *_vals_arr;
index_t  _vals_arr_size;

typedef struct {
	index_t row;
	index_t col;
	value_t val;
} coo_tuple_t;

typedef vector<index_t> index_1d_t;
typedef vector<index_1d_t> index_2d_t;
typedef vector<index_2d_t> index_3d_t;
typedef vector<value_t> value_1d_t;
typedef vector<value_1d_t> value_2d_t;
typedef vector<value_2d_t> value_3d_t;

typedef vector<coo_tuple_t> tuple_1d_t; 
typedef vector<tuple_1d_t> tuple_2d_t; 
typedef vector<tuple_2d_t> tuple_3d_t; 

struct slice_t_tag {
	int global_tid;
	int panel_id;
	index_t nnz;
	index_t n_vec;
	index_t start_nnz;
	index_t end_nnz;
	index_t vec_write_base;
	index_t start_vec;
	index_t end_vec;
	index_t first_row;
	index_t last_row;
	index_t overflow_row;
	index_t n_colblk;
	index_t n_rowblk;
	index_1d_t row_arr;
	index_1d_t col_arr;
	value_1d_t val_arr;

	index_3d_t row_blocks;
	index_3d_t col_blocks;
	value_3d_t val_blocks;

	tuple_3d_t blocks;

	index_2d_t block_nnz_cnt;

	index_1d_t tmp_veceor;
	index_1d_t tmp_row_arr;
	vector<uint8_t> tmp_scan_mask;
};
typedef struct slice_t_tag slice_t;

template<typename T>
struct reverse_weight_sorter
{
public:
reverse_weight_sorter(vector<T>& weights) : _weights(weights) { }
  bool operator()(const int i, const int j) {
    if (_weights[i] > _weights[j]) return true;
    return 0;
  }
private:
	vector<T>& _weights;
};
struct rowcol_sorter
{
public:
  bool operator()(const coo_tuple_t& i, const coo_tuple_t& j) {
    if (i.row < j.row) return true;
    if (i.row > j.row) return false;
    if (i.col < j.col) return true;
    if (i.col > j.col) return false;
    return 0;
  }
};
struct colrow_sorter
{
public:
  bool operator()(const coo_tuple_t& i, const coo_tuple_t& j) {
    if (i.col < j.col) return true;
    if (i.col > j.col) return false;
    if (i.row < j.row) return true;
    if (i.row > j.row) return false;
    return 0;
  }
};

typedef struct {
	int id;
	int base_tid;
	int num_threads;
	index_t nnz;
	index_t start_nnz;
	index_t end_nnz;
	
	vector<coo_tuple_t> coo;
	vector<slice_t> slices;
} panel_t;

void partition(panel_t& panel);
void sort_vertical_partition(panel_t& panel);

colrow_sorter _panel_sorter;
rowcol_sorter _slice_sorter;

};



// implementation

template<typename IndexType, typename ValueType>
void vhcc_matrix<IndexType, ValueType>
	::partition(panel_t& panel)
{
	vector<slice_t>& slices  = panel.slices;
	vector<coo_tuple_t>& coo = panel.coo;
	int nnz = panel.nnz;
	int num_threads = panel.num_threads;

        // sort from horizonal from the vertical
	stable_sort(panel.coo.begin(), panel.coo.end(), _slice_sorter);

	slices.resize(panel.num_threads);
	if (panel.num_threads == 0)
		return;

	index_t	nnz_per_thread = (nnz+num_threads-1) / num_threads;

	index_t nnz_remain = nnz;
	index_t cur_index = 0;
	for (int i = 0; i < num_threads; ++i) {
		slice_t& slice   = slices[i];
		slice.global_tid = panel.base_tid + i;
		slice.panel_id   = panel.id;
		
		if (nnz_remain >= nnz_per_thread) {
			slice.nnz   = nnz_per_thread;
			nnz_remain -= nnz_per_thread;
		} else {
			slice.nnz  = nnz_remain;
			nnz_remain = 0;
		}
		slice.n_vec     = (slice.nnz+VECLEN-1) / VECLEN;
		slice.start_nnz = cur_index;
		slice.end_nnz   = cur_index + slice.nnz;
		cur_index      += slice.nnz;
	}

#pragma omp parallel for
	for (int i = 0; i < num_threads; ++i) {
		slice_t& slice = slices[i];

		slice.row_arr  = index_1d_t(slice.n_vec*VECLEN);
		slice.col_arr  = index_1d_t(slice.n_vec*VECLEN);
		slice.val_arr  = value_1d_t(slice.n_vec*VECLEN);

		index_t tstart  = slice.start_nnz;
		index_t tend    = slice.end_nnz;
		slice.first_row = coo[tstart].row;
		slice.last_row  = tend == tstart ? slice.first_row : coo[tend-1].row;
		slice.overflow_row = _extended_rows + slice.global_tid * VECLEN;
		
		index_t drows = slice.last_row - slice.first_row;
		index_t n_colblk = (_clength != -1) ? (_num_cols+_clength-1)/_clength : 1;
		index_t n_rowblk = (_rlength != -1) ? (drows+_rlength-1)/_rlength + 1 : 1;
		slice.n_colblk = n_colblk;
		slice.n_rowblk = n_rowblk;
		slice.blocks = tuple_3d_t(n_rowblk, tuple_2d_t(n_colblk, tuple_1d_t()));

		slice.block_nnz_cnt = index_2d_t(n_rowblk, index_1d_t(n_colblk, 0));

		for (index_t j = tstart; j < tend; ++j) {
			index_t row = coo[j].row;
			index_t col = coo[j].col;

			index_t rblk = (_rlength != -1) ? (row-slice.first_row)/_rlength : 0;
			index_t cblk = (_clength != -1) ? col/_clength : 0;

			slice.block_nnz_cnt[rblk][cblk] += 1;
		}
		for (int p = 0; p < n_rowblk; ++p) {
			for (int q = 0; q < n_colblk; ++q) {
				int reserve = slice.block_nnz_cnt[p][q];
				slice.blocks[p][q].reserve(reserve);
			}
		}
		for (index_t j = tstart; j < tend; ++j) {
			index_t row = coo[j].row;
			index_t col = coo[j].col;
			value_t val = coo[j].val;

			index_t shifted_row;
			if (row == slice.last_row)
				shifted_row = slice.overflow_row;
			else
				shifted_row = row;

			index_t rblk = (_rlength != -1) ? (row-slice.first_row)/_rlength : 0;
			index_t cblk = (_clength != -1) ? col/_clength : 0;

			coo_tuple_t tuple = { shifted_row, col, val };
			slice.blocks[rblk][cblk].push_back(tuple);
		}
		
		index_t ptr = 0;
		for (index_t c = 0; c < slice.n_colblk; ++c) {
			for (index_t r = 0; r < slice.n_rowblk; ++r) {
				rowcol_sorter sorter;
				stable_sort(slice.blocks[r][c].begin(), slice.blocks[r][c].end(), sorter);
				for (index_t k = 0; k < slice.blocks[r][c].size(); ++k) {
					slice.row_arr[ptr] = slice.blocks[r][c][k].row;
					slice.col_arr[ptr] = slice.blocks[r][c][k].col;
					slice.val_arr[ptr] = slice.blocks[r][c][k].val;
					ptr++;
				}
			}
		}
		int pad = slice.n_vec * VECLEN - slice.nnz;
		for (int p = 0; p < pad; ++p) {
			slice.row_arr[ptr + p] = slice.row_arr[ptr-1];
			slice.col_arr[ptr + p] = slice.col_arr[ptr-1];
			slice.val_arr[ptr + p] = 0;
		}
	}
	

#pragma omp parallel for
	for (int i = 0; i < num_threads; ++i) {
		slice_t& slice = slices[i];

		slice.tmp_veceor.reserve(_num_rows);
		slice.tmp_row_arr.reserve(_num_rows);
		slice.tmp_scan_mask.reserve(_num_rows * 4);

		for (index_t v = 0; v < slice.n_vec; ++v) {
			index_t vstart = v*VECLEN;
			index_t vend   = (v+1)*VECLEN;
			index_t vlen   = vend - vstart;

			index_t tmprow1[VECLEN], tmprow2[VECLEN], eor[VECLEN];
			tmprow1[0:vlen] = slice.row_arr.data()[vstart:vlen];

			if (vend == slice.n_vec*VECLEN) {
				tmprow2[0:vlen-1] = slice.row_arr.data()[vstart+1:vlen-1];
				tmprow2[vlen-1] = slice.row_arr[vend-1] + 1;
			} else {
				tmprow2[0:vlen] = slice.row_arr.data()[vstart+1:vlen];
			}
			eor[:] = (tmprow2[:] - tmprow1[:]) != 0;

			int cnt = __sec_reduce_add(eor[:]);
			bool is_eor = cnt > 0;
			
			if (is_eor) {
				slice.tmp_veceor.push_back(v);

				for (int i = 0; i < VECLEN; ++i)
					if (eor[i] == 1)
						slice.tmp_row_arr.push_back(tmprow1[i]);
			
				mask_t mask1, mask2, mask3, maskwr, tmask, m;
				mask1 = mask2 = mask3 = maskwr = 0;
				for (int i = 0; i < VECLEN; ++i) maskwr |= (eor[i] << i);
				tmask = maskwr << 1;
				mask1 = (~tmask) & 0xAA;
				m = tmask & 0xCC;
				mask2 = (~(m | m<<1)) & 0xCC;
				m = tmask & 0xF0;
				mask3 = (~(m | m<<1 | m <<2 | m << 3)) & 0xF0;
				slice.tmp_scan_mask.push_back(mask1);
				slice.tmp_scan_mask.push_back(mask2);
				slice.tmp_scan_mask.push_back(mask3);
				slice.tmp_scan_mask.push_back(maskwr);
			}
		}
	}

}

template<typename IndexType, typename ValueType>
void vhcc_matrix<IndexType, ValueType>
	::convert(int num_threads, int threads_per_core, int num_panels, int rlength, int clength, char* filename)
{
	vector<index_t> row_ptr(_num_rows+1, 0);
	index_t r = 0;
	row_ptr[0] = 0;
    
        // do the row of CSR format
	for (index_t i = 0; i < _num_entries; ++i) {
		index_t ridx = _row_idx[i] + 1 - 1;
		if (r != ridx) {
			for (index_t p = r+1; p < ridx; ++p)
				row_ptr[p] = i;
			r = ridx;
			row_ptr[r] = i;
		}
	}
    
        // do the last 0-nnz rows of CSR format
	for (index_t p = r+1; p < _num_rows; ++p)
		row_ptr[p] = _num_entries;
	row_ptr[_num_rows] = _num_entries;


  double t_pre = microtime();


	index_t max_nnzpr = -1;
	index_t min_nnzpr = _num_entries;
	double std_nnzpr = 0;
	double ave_nnzpr = double(_num_entries)/_num_rows;
#pragma omp parallel for reduction(max: max_nnzpr) reduction(min: min_nnzpr) reduction(+: std_nnzpr)
	for (index_t i = 0; i < _num_rows; ++i) {
		index_t ncol = row_ptr[i+1] - row_ptr[i];
		min_nnzpr  = ncol < min_nnzpr ? ncol : min_nnzpr;
		max_nnzpr  = ncol > max_nnzpr ? ncol : max_nnzpr;
		std_nnzpr += (ncol - ave_nnzpr)*(ncol - ave_nnzpr);
	}
	std_nnzpr = sqrt(std_nnzpr/_num_rows);

	vector<coo_tuple_t> coo(_num_entries);
#pragma omp parallel for
	for (int i = 0; i < _num_entries; ++i) {
		coo_tuple_t tmp_tuple;
		tmp_tuple.row = _row_idx[i];
		tmp_tuple.col = _col_idx[i];
		tmp_tuple.val = _vals[i];
		coo[i] = tmp_tuple;
	}
	if (rlength < -1 || clength < -1) {
		printf("negtive number invalid.\n");
		exit(1);
	}
	_rlength       = rlength;
	_clength       = clength;
	_num_threads   = num_threads;
	if (num_threads % threads_per_core != 0) {
		printf("Expect num_threads to be divisible by %d\n", threads_per_core);
		exit(1);
	}
	_num_cores     = (num_threads + threads_per_core-1) / threads_per_core;
	_num_panels    = num_panels;
	_nnz_per_panel = (_num_entries+_num_panels-1) / _num_panels;
	_extended_rows = (_num_rows + VECLEN-1) / VECLEN * VECLEN;
	if (_num_threads % _num_panels != 0) {
		printf("Num_threads not divisible by panels %d mod %d\n", _num_threads, _num_panels);
		exit(1);
	}
	int threads_per_panel = _num_threads / _num_panels;
        std::cout<<"sorting............."<<endl;
	stable_sort(coo.begin(), coo.end(), _panel_sorter);
	std::cout<<"sorting.................end "<<endl;
	vector<panel_t> panels(_num_panels);
	index_t thr_remain = _num_threads;
	index_t nnz_remain = _num_entries;
	index_t cur_index  = 0;
	index_t base_tid   = 0;



	for (int i = 0; i < _num_panels; ++i) {
		panel_t& panel = panels[i];
		panel.id = i;

		if (thr_remain >= threads_per_panel) {
			panel.num_threads = threads_per_panel;
			thr_remain -= threads_per_panel;
		} else {
			panel.num_threads  = thr_remain;
			thr_remain = 0;
		}
		panel.base_tid = base_tid;
		base_tid      += panel.num_threads;
		
		if (nnz_remain >= _nnz_per_panel) {
			panel.nnz = _nnz_per_panel;
			nnz_remain -= _nnz_per_panel;
		} else {
			panel.nnz  = nnz_remain;
			nnz_remain = 0;
		}

		panel.start_nnz = cur_index;
		panel.end_nnz   = cur_index + panel.nnz;
		cur_index      += panel.nnz;
	}

#pragma omp parallel for
	for (int c = 0; c < _num_panels; ++c) {
		panel_t& panel = panels[c];

		panel.coo.resize(panel.nnz);
		for (int k = 0; k < panel.nnz; ++k) {
			panel.coo[k] = coo[panel.start_nnz + k];
		}
		
		partition(panel);
	}


	index_t veceor_size = 0;
	index_t row_arr_size = 0;
	index_t scan_mask_size = 0;
#pragma omp parallel for reduction(+: veceor_size, row_arr_size, scan_mask_size)
	for (int j = 0; j < _num_panels; ++j) {
		for (int i = 0; i < panels[j].num_threads; ++i) {
			slice_t& slice = panels[j].slices[i];

			veceor_size    += slice.tmp_veceor.size();
			row_arr_size   += slice.tmp_row_arr.size();
			scan_mask_size += slice.tmp_scan_mask.size();
		}
	}

	index_t num_vectors = 0;
	for (int j = 0; j < _num_panels; ++j) {
		for (int i = 0; i < panels[j].num_threads; ++i) {
			slice_t& slice       = panels[j].slices[i];
			slice.vec_write_base = num_vectors * VECLEN;
			slice.start_vec      = num_vectors;
			slice.end_vec        = num_vectors + slice.n_vec;
			num_vectors         += slice.n_vec;
		}
	}

	_num_vectors = num_vectors;
	_pad_entries = _num_vectors * VECLEN;
	_pad_rows    = _extended_rows + _num_threads * VECLEN;
	_pad_cols    = _num_cols;
	
	_thr_info_size = _num_threads;
	_thr_info      = (thr_info_t *)MALLOC(_thr_info_size * sizeof(thr_info_t));
	_col_arr_size  = _num_vectors * VECLEN;
	_vals_arr_size = _num_vectors * VECLEN;
	_col_arr       = (index_t *)MALLOC(_col_arr_size * sizeof(index_t));
	_vals_arr      = (value_t *)MALLOC(_vals_arr_size * sizeof(value_t));
	_tmp_row_arr   = (index_t *)MALLOC(_num_vectors*VECLEN * sizeof(index_t));

	_veceor_size    = veceor_size;
	_row_arr_size   = row_arr_size;
	_scan_mask_size = scan_mask_size;
	_veceor_ptr     = (index_t *)MALLOC(_veceor_size * sizeof(index_t));
	_row_arr        = (index_t *)MALLOC(_row_arr_size * sizeof(index_t));
	_scan_mask      = (uint8_t *)MALLOC(_scan_mask_size * sizeof(uint8_t));

#pragma omp parallel for	
	for (int j = 0; j < _num_panels; ++j) {
		for (int i = 0; i < panels[j].num_threads; ++i) {
			slice_t& slice = panels[j].slices[i];

			std::copy(slice.col_arr.begin(), slice.col_arr.end(), &_col_arr[slice.vec_write_base]);
			std::copy(slice.val_arr.begin(), slice.val_arr.end(), &_vals_arr[slice.vec_write_base]);
			// for debugging use
			std::copy(slice.row_arr.begin(), slice.row_arr.end(), &_tmp_row_arr[slice.vec_write_base]);
		}
	}
	
	index_t v_accum = 0;
	index_t vbase = 0;
	index_t rbase = 0;
	index_t sbase = 0;
	for (int j = 0; j < _num_panels; ++j) {
		for (int i = 0; i < panels[j].num_threads; ++i) {
			slice_t& slice = panels[j].slices[i];
			int tid = slice.global_tid;

			_thr_info[tid].vbase        = vbase;
			_thr_info[tid].sbase        = sbase;
			_thr_info[tid].rbase        = rbase;
			_thr_info[tid].last_row     = slice.last_row;
			_thr_info[tid].overflow_row = slice.overflow_row;
			_thr_info[tid].start_vec    = slice.start_vec;
			_thr_info[tid].end_vec      = slice.end_vec;
			_thr_info[tid].panel_id     = slice.panel_id;
			_thr_info[tid].merge_start  = 0;
			_thr_info[tid].merge_end    = 0;
		

			for (index_t v = 0; v < slice.tmp_veceor.size(); ++v) {
				_veceor_ptr[vbase + v] = slice.tmp_veceor[v] + v_accum;
			}
			v_accum += slice.n_vec;
		
			std::copy(slice.tmp_row_arr.begin(), slice.tmp_row_arr.end(), &_row_arr[rbase]);
			std::copy(slice.tmp_scan_mask.begin(), slice.tmp_scan_mask.end(), &_scan_mask[sbase]);

			vbase += slice.tmp_veceor.size();
			rbase += slice.tmp_row_arr.size();
			sbase += slice.tmp_scan_mask.size();
		
		}
	}

	_panel_info = (panel_info_t *)MALLOC(_num_panels * sizeof(panel_info_t));
	for (int i = 0; i < _num_panels; ++i) {
		value_t *tmp_result = (value_t *)MALLOC(_pad_rows * sizeof(value_t));
		memset(tmp_result, 0, _pad_rows * sizeof(value_t));

		_panel_info[i].tmp_result = tmp_result;
	}

	int workers_per_core        = threads_per_core;
	int n_workers               = _num_cores * workers_per_core;
	index_t num_yvec            = _extended_rows / VECLEN;
	index_t num_yvec_per_thread = (num_yvec + n_workers-1) / n_workers;
	for (int i = 0; i < n_workers; ++i) {
		index_t merge_start      = i * num_yvec_per_thread*VECLEN;
		index_t merge_end        = merge_start + num_yvec_per_thread*VECLEN > _extended_rows ? _extended_rows : merge_start + num_yvec_per_thread*VECLEN;

		int coreid = i / workers_per_core;
		int lthrid = i % workers_per_core;
		int globid = i;
		_thr_info[globid].merge_start = merge_start;
		_thr_info[globid].merge_end   = merge_end;
	}
  std::cout<<"============="<<endl;
//  std::cout<<"The Pre-processing Time of VHCC is  "<<filename<<" "<< microtime() - t_pre<<endl;
  std::cout<<"The Pre-processing(CSR->VHCC)  Time of VHCC  is "<< microtime() - t_pre<<" seconds.    [file: "<<filename<<"] [threads: "<<num_threads<<"] [numPanels: "<<num_panels<<"]"<<endl;
  std::cout<<"============="<<endl;


	printf("Matrix %d x %d, nnz %d\n", _num_rows, _num_cols, _num_entries);
	printf("Number of threads: %d\n", _num_threads);
	printf("Number of cores: %d\n", _num_cores);
	printf("Number of panels: %d\n", _num_panels);
	printf("Threads per core: %d\n", threads_per_core);
	printf("\n");

}


