/* Differential fixture for the three SelectionVector C-API functions
 * (D-DR-SELECT-K-SV-3). Uses only duckdb.h declarations. Build it once
 * against the real libduckdb and once against the adapter; the two
 * processes must print the same bytes. Prints no addresses. */
#include "duckdb.h"
#include <stdio.h>
#include <string.h>

static uint64_t mix(uint64_t x) {
	x += 0x9E3779B97F4A7C15ull;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
	return x ^ (x >> 31);
}

static uint64_t checksum(const sel_t *p, idx_t n) {
	uint64_t h = 1469598103934665603ull;
	for (idx_t i = 0; i < n; i++) {
		h = (h ^ p[i]) * 1099511628211ull;
	}
	return h;
}

int main(void) {
	static const idx_t sizes[] = {0, 1, 64, 256, 2048, 65536};
	for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
		idx_t n = sizes[s];
		duckdb_selection_vector sel = duckdb_create_selection_vector(n);
		sel_t *p = duckdb_selection_vector_get_data_ptr(sel);
		printf("size=%llu handle_nonnull=%d ptr_nonnull=%d", (unsigned long long)n, sel != NULL, p != NULL);
		/* permuted, with repeats: every third entry repeats its predecessor */
		for (idx_t i = 0; i < n; i++) {
			p[i] = (i % 3 == 2) ? p[i - 1] : (sel_t)(mix(i ^ (n << 20)) % (n ? n : 1));
		}
		sel_t *q = duckdb_selection_vector_get_data_ptr(sel);
		printf(" same_ptr=%d sum_a=%016llx", p == q, (unsigned long long)checksum(q, n));
		/* mutation through the first pointer is seen through a later one */
		if (n > 0) {
			p[0] = p[0] + 1;
			p[n - 1] = 0xFFFFFFFFu;
		}
		sel_t *r = duckdb_selection_vector_get_data_ptr(sel);
		printf(" sum_b=%016llx last=%llu", (unsigned long long)checksum(r, n),
		       (unsigned long long)(n ? r[n - 1] : 0));
		/* repeats and order are kept as written */
		idx_t repeats = 0, descents = 0;
		for (idx_t i = 1; i + 1 < n; i++) {
			repeats += r[i] == r[i - 1];
			descents += r[i] < r[i - 1];
		}
		printf(" repeats=%llu descents=%llu\n", (unsigned long long)repeats, (unsigned long long)descents);
		duckdb_destroy_selection_vector(sel);
	}
	/* two live selections do not share storage */
	duckdb_selection_vector a = duckdb_create_selection_vector(256);
	duckdb_selection_vector b = duckdb_create_selection_vector(256);
	sel_t *pa = duckdb_selection_vector_get_data_ptr(a);
	sel_t *pb = duckdb_selection_vector_get_data_ptr(b);
	for (idx_t i = 0; i < 256; i++) {
		pa[i] = (sel_t)i;
		pb[i] = (sel_t)(255 - i);
	}
	printf("distinct=%d a=%016llx b=%016llx\n", pa != pb, (unsigned long long)checksum(pa, 256),
	       (unsigned long long)checksum(pb, 256));
	/* a pointer stays valid while the other selection is destroyed */
	duckdb_destroy_selection_vector(b);
	printf("a_after_b_destroyed=%016llx\n", (unsigned long long)checksum(pa, 256));
	duckdb_destroy_selection_vector(a);
	/* churn, and NULL destroy */
	uint64_t acc = 0;
	for (int i = 0; i < 2000; i++) {
		duckdb_selection_vector c = duckdb_create_selection_vector((idx_t)(i % 97));
		sel_t *pc = duckdb_selection_vector_get_data_ptr(c);
		pc[0] = (sel_t)i;
		acc += pc[0];
		duckdb_destroy_selection_vector(c);
	}
	duckdb_destroy_selection_vector(NULL);
	printf("churn=%llu null_destroy=ok\n", (unsigned long long)acc);
	return 0;
}
