// Procedural adjacency vs enumerated edges: A_n = [[1,1],[1,0]]^(Kronecker n)
// g++ -O3 -std=c++17 -march=native -o fractal_bench main.cpp
// No third-party dependencies.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using Counts = std::vector<uint64_t>;

uint64_t ipow(uint64_t base, int exp) {
    uint64_t out = 1;
    for (int i = 0; i < exp; ++i) out *= base;
    return out;
}

// The explicit baseline visits EVERY edge (even though it streams rather
// than storing the full edge list, which favors the baseline on memory).
// Each selected edge is binned into a screen-space pixel.
void visit_edges(int bit, uint32_t u, uint32_t v,
                 int shift, unsigned side, Counts& pixels, uint64_t value) {
    if (bit < 0) {
        const unsigned row = u >> shift;
        const unsigned col = v >> shift;
        pixels[size_t(row) * side + col] += value;
        return;
    }
    const uint32_t one = uint32_t(1) << bit;
    visit_edges(bit - 1, u,       v,       shift, side, pixels, value); // 00
    visit_edges(bit - 1, u,       v | one, shift, side, pixels, value); // 01
    visit_edges(bit - 1, u | one, v,       shift, side, pixels, value); // 10
}

void render_explicit(int n, int k, Counts& pixels) {
    std::fill(pixels.begin(), pixels.end(), 0);
    visit_edges(n - 1, 0, 0, n - k, 1u << k, pixels, 1);
}

// Procedural renderer A: evaluate a simple O(1) adjacency predicate for
// each output pixel. Below native resolution this is exact OR aggregation:
// each active k-bit prefix has exactly 3^(n-k) child edges.
void render_rule_pixels(int n, int k, Counts& pixels) {
    const unsigned side = 1u << k;
    const uint64_t count = ipow(3, n - k);
    for (unsigned u = 0; u < side; ++u) {
        for (unsigned v = 0; v < side; ++v) {
            pixels[size_t(u)*side + v] = ((u & v) == 0) ? count : 0;
        }
    }
}

// Procedural renderer B: enumerate only the VISIBLE k-depth pattern,
// not the graph's actual n-depth edges. Constant work once n >= k.
void render_rule_sparse(int n, int k, Counts& pixels) {
    std::fill(pixels.begin(), pixels.end(), 0);
    visit_edges(k - 1, 0, 0, 0, 1u << k, pixels, ipow(3, n - k));
}

// Pixel-perfect viewport at the actual n-bit graph depth (NOT aggregation).
// This demonstrates that zooming into depth=26 still needs only 512^2
// predicate evaluations, irrespective of the full graph's edge count.
void render_zoom_exact(int n, Counts& pixels) {
    constexpr unsigned side = 512;
    const uint32_t start_u = n == 9 ? 0 : (1u << (n - 1));
    const uint32_t start_v = n <= 10 ? 0 : (1u << (n - 2));
    for (unsigned i = 0; i < side; ++i) {
        for (unsigned j = 0; j < side; ++j) {
            const uint32_t u = start_u + i;
            const uint32_t v = start_v + j;
            pixels[size_t(i) * side + j] = ((u & v) == 0) ? 1 : 0;
        }
    }
}

uint64_t checksum(const Counts& pixels) {
    // Hash every entry so rendering work is observable, but time hash separately.
    uint64_t h = 14695981039346656037ULL;
    for (uint64_t v : pixels) { h ^= v; h *= 1099511628211ULL; }
    return h;
}

void save_pgm(const std::string& filename, const Counts& pixels, unsigned side) {
    std::ofstream file(filename, std::ios::binary);
    file << "P5\n" << side << ' ' << side << "\n255\n";
    for (uint64_t val : pixels) {
        const unsigned char shade = (val > 0) ? 255 : 0;
        file.write(reinterpret_cast<const char*>(&shade), 1);
    }
}

void save_zoom_pgm(const std::string& filename, int n, unsigned bits,
                   uint32_t first_u, uint32_t first_v) {
    const unsigned side = 1u << bits;
    const uint64_t limit = uint64_t(1) << n;
    if (uint64_t(first_u) + side > limit || uint64_t(first_v) + side > limit) {
        std::cerr << "Zoom ROI out of bounds\n"; std::exit(1);
    }
    std::ofstream file(filename, std::ios::binary);
    file << "P5\n" << side << ' ' << side << "\n255\n";
    for (unsigned i = 0; i < side; ++i) {
        for (unsigned j = 0; j < side; ++j) {
            const unsigned char shade = (((first_u+i) & (first_v+j)) == 0) ? 255 : 0;
            file.write(reinterpret_cast<const char*>(&shade), 1);
        }
    }
}

template<class Fn>
double bench_ms(Fn fn, int repetitions) {
    double best = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < repetitions; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(t1-t0).count();
        best = std::min(best, elapsed);
    }
    return best;
}

int main(int argc, char** argv) {
    const int max_explicit = argc > 1 ? std::atoi(argv[1]) : 15;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
    const int max_screen_bits = argc > 3 ? std::atoi(argv[3]) : 9;
    if (max_explicit < 0 || max_explicit > 26 || repeats < 1 ||
        max_screen_bits < 1 || max_screen_bits > 11) {
        std::cerr << "Usage: " << argv[0]
                  << " [max_explicit_depth 0..26=15] [repeats >=1=5]"
                  << " [screen_bits 1..11=9]\n";
        return 2;
    }
    std::cout << "depth,vertices_per_side,edges,screen_side,screen_pixels,"
                 "explicit_ms,predicate_ms,rule_sparse_ms,zoom_exact_ms,"
                 "pixel_equal,sparse_equal,checksum\n";

    const int depths[] = {4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26};
    for (int n : depths) {
        const int k = std::min(n, max_screen_bits);
        const unsigned side = 1u << k;
        Counts explicit_img(size_t(side)*side, 0);
        Counts predicate_img(size_t(side)*side, 0);
        Counts sparse_img(size_t(side)*side, 0);

        const double pms = bench_ms([&]() {render_rule_pixels(n,k,predicate_img);}, repeats);
        const double sms = bench_ms([&]() {render_rule_sparse(n,k,sparse_img);}, repeats);
        const uint64_t expected = ipow(3, n);
        const uint64_t observed = std::accumulate(predicate_img.begin(), predicate_img.end(), uint64_t(0));
        const bool sparse_equal = (sparse_img == predicate_img);
        if (!sparse_equal || observed != expected) {
            std::cerr << "FAIL: mathematical invariant at depth " << n << '\n';
            return 1;
        }
        double zms = -1;
        if (n >= 9) {
            Counts zoom_img(512u*512u, 0);
            zms = bench_ms([&]() {render_zoom_exact(n,zoom_img);}, repeats);
            // Independently check the zoom against the membership formula.
            const uint32_t ox = n == 9 ? 0 : (1u << (n - 1));
            const uint32_t oy = n <= 10 ? 0 : (1u << (n - 2));
            for (unsigned i : {0u, 1u, 117u, 255u, 511u}) {
                for (unsigned j : {0u, 17u, 256u, 511u}) {
                    if (zoom_img[size_t(i)*512+j] != (((ox+i)&(oy+j))==0)) {
                        std::cerr << "FAIL: exact zoom at depth " << n << '\n';
                        return 1;
                    }
                }
            }
        }
        bool pixel_equal = false;
        double ems = -1;
        if (n <= max_explicit) {
            const int explicit_reps = n >= 15 ? 1 : repeats;
            ems = bench_ms([&]() {render_explicit(n,k,explicit_img);}, explicit_reps);
            pixel_equal = (explicit_img == predicate_img);
            if (!pixel_equal) {
                std::cerr << "FAIL: explicit != procedural at depth " << n << '\n';
                return 1;
            }
        }
        const uint64_t h = checksum(predicate_img);
        std::cout << n << ',' << (uint64_t(1)<<n) << ',' << expected << ','
                  << side << ',' << uint64_t(side)*side << ',';
        if (ems < 0) std::cout << "NA";
        else std::cout << std::fixed << std::setprecision(6) << ems;
        std::cout << ',' << std::fixed << std::setprecision(6) << pms
                  << ',' << sms << ',';
        if (zms < 0) std::cout << "NA";
        else std::cout << zms;
        std::cout << ',' << (n <= max_explicit ? "true" : "not_run")
                  << ',' << (sparse_equal ? "true" : "false")
                  << ',' << h << '\n';
        if (n == 9) save_pgm("depth9_native.pgm", predicate_img, side);
        if (n == 12) save_pgm("depth12_full_view.pgm", predicate_img, side);
        if (n == 26) save_pgm("depth26_full_view.pgm", predicate_img, side);
    }
    save_zoom_pgm("depth26_zoom_512.pgm", 26, 9, 1u << 17, 1u << 18);
    return 0;
}
