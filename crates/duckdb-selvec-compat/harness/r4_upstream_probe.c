/* R4 / R7 reconnaissance against the REAL DuckDB only (D-DR-SELECT-K-SV-3).
 * Records what duckdb_slice_vector and duckdb_vector_copy_sel do with
 * permutations, repeats, nested selections, NULLs and a destroyed selection.
 * Not run against the adapter: it does not provide these two functions. */
#include "duckdb.h"
#include <stdio.h>

#define N 8

static void dump(const char *name, duckdb_vector v, idx_t n) {
	/* logical values: copy through an identity selection into a flat vector */
	duckdb_logical_type t = duckdb_create_logical_type(DUCKDB_TYPE_INTEGER);
	duckdb_data_chunk out = duckdb_create_data_chunk(&t, 1);
	duckdb_vector dst = duckdb_data_chunk_get_vector(out, 0);
	duckdb_selection_vector id = duckdb_create_selection_vector(n);
	sel_t *ip = duckdb_selection_vector_get_data_ptr(id);
	for (idx_t i = 0; i < n; i++) {
		ip[i] = (sel_t)i;
	}
	duckdb_vector_copy_sel(v, dst, id, n, 0, 0);
	int32_t *d = (int32_t *)duckdb_vector_get_data(dst);
	uint64_t *val = duckdb_vector_get_validity(dst);
	printf("%s:", name);
	for (idx_t i = 0; i < n; i++) {
		if (val && !duckdb_validity_row_is_valid(val, i)) {
			printf(" NULL");
		} else {
			printf(" %d", d[i]);
		}
	}
	printf("\n");
	duckdb_destroy_selection_vector(id);
	duckdb_destroy_data_chunk(&out);
	duckdb_destroy_logical_type(&t);
}

int main(void) {
	printf("library=%s vector_size=%llu\n", duckdb_library_version(), (unsigned long long)duckdb_vector_size());
	duckdb_logical_type t = duckdb_create_logical_type(DUCKDB_TYPE_INTEGER);
	duckdb_data_chunk chunk = duckdb_create_data_chunk(&t, 1);
	duckdb_vector v = duckdb_data_chunk_get_vector(chunk, 0);
	int32_t *base = (int32_t *)duckdb_vector_get_data(v);
	for (int i = 0; i < N; i++) {
		base[i] = 100 + i;
	}
	duckdb_vector_ensure_validity_writable(v);
	duckdb_validity_set_row_invalid(duckdb_vector_get_validity(v), 3);
	duckdb_data_chunk_set_size(chunk, N);
	dump("base", v, N);

	/* permutation with repeats, including the NULL row twice */
	duckdb_selection_vector s1 = duckdb_create_selection_vector(6);
	sel_t *p1 = duckdb_selection_vector_get_data_ptr(s1);
	sel_t a[6] = {7, 3, 3, 0, 5, 7};
	for (int i = 0; i < 6; i++) {
		p1[i] = a[i];
	}
	duckdb_slice_vector(v, s1, 6);
	dump("slice1", v, 6);
	int32_t *after = (int32_t *)duckdb_vector_get_data(v);
	printf("slice1 data_ptr_same_as_base=%d raw0=%d raw7=%d (raw = duckdb_vector_get_data, unselected)\n",
	       after == base, after[0], after[7]);

	/* writing the selection AFTER the slice: does the vector follow? */
	p1[0] = 1;
	dump("slice1_after_sel_write", v, 6);

	/* destroying the selection after the slice */
	duckdb_destroy_selection_vector(s1);
	dump("slice1_after_sel_destroy", v, 6);

	/* nested: a second selection addresses the result of the first */
	duckdb_selection_vector s2 = duckdb_create_selection_vector(4);
	sel_t *p2 = duckdb_selection_vector_get_data_ptr(s2);
	sel_t b[4] = {5, 5, 1, 4};
	for (int i = 0; i < 4; i++) {
		p2[i] = b[i];
	}
	duckdb_slice_vector(v, s2, 4);
	dump("slice2_nested", v, 4);
	duckdb_destroy_selection_vector(s2);

	/* copy_sel with offsets: count is an END index, not a length */
	duckdb_data_chunk c2 = duckdb_create_data_chunk(&t, 1);
	duckdb_vector src = duckdb_data_chunk_get_vector(c2, 0);
	int32_t *sb = (int32_t *)duckdb_vector_get_data(src);
	for (int i = 0; i < N; i++) {
		sb[i] = 200 + i;
	}
	duckdb_data_chunk c3 = duckdb_create_data_chunk(&t, 1);
	duckdb_vector dst = duckdb_data_chunk_get_vector(c3, 0);
	int32_t *db = (int32_t *)duckdb_vector_get_data(dst);
	for (int i = 0; i < N; i++) {
		db[i] = -1;
	}
	duckdb_selection_vector s3 = duckdb_create_selection_vector(5);
	sel_t *p3 = duckdb_selection_vector_get_data_ptr(s3);
	sel_t c[5] = {6, 6, 2, 0, 4};
	for (int i = 0; i < 5; i++) {
		p3[i] = c[i];
	}
	duckdb_vector_copy_sel(src, dst, s3, 5, 1, 2);
	printf("copy_sel(count=5,src_offset=1,dst_offset=2):");
	for (int i = 0; i < N; i++) {
		printf(" %d", db[i]);
	}
	printf("\n");
	duckdb_destroy_selection_vector(s3);
	duckdb_destroy_data_chunk(&c3);
	duckdb_destroy_data_chunk(&c2);
	duckdb_destroy_data_chunk(&chunk);
	duckdb_destroy_logical_type(&t);
	return 0;
}
