// Standalone executable over the same engine as the ctypes library (gridpath.cpp).
//
// It exists for X9, the engine benchmark: the grid is loaded ONCE from a .npy file and every
// query of a batch is searched in this process, so the comparison against the in-process
// ctypes engine measures the cost of the process boundary and of the load, not a different
// algorithm. The search itself is the very same gp_dijkstra function.
//
// usage:
//   gridpath_exe GRID.npy MASK.npy|- QUERIES.txt KIND A B C HAS_PEN I_MAX PEN HAS_CAP CAP
//                LEN_STRAIGHT LEN_DIAG REPEATS [DIST_OUT.bin]
// QUERIES.txt: one "start target" pair of flat node ids per line.
// Output, one JSON object per line on stdout:
//   {"load_ms": ...}                                  first line
//   {"query": k, "start": s, "target": t, "cost": c, "expanded": e, "pushed": p,
//    "reached": r, "heap_bytes": h, "search_ms": [ ... one per repeat ... ]}
// With DIST_OUT.bin the distances of the LAST query are written as raw float64 (tests).
#include "gridpath.cpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace {

struct Array {
    std::vector<char> bytes;
    std::string descr;
    int64_t rows = 0, cols = 0;
};

// Minimal .npy reader: version 1/2/3, C order, 2-D, little-endian float64 or uint8/bool.
bool read_npy(const char* path, Array* out, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { *error = std::string("cannot open ") + path; return false; }
    char magic[6];
    in.read(magic, 6);
    if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) { *error = "not a .npy file"; return false; }
    unsigned char version[2];
    in.read(reinterpret_cast<char*>(version), 2);
    uint32_t header_len = 0;
    if (version[0] == 1) {
        unsigned char b[2];
        in.read(reinterpret_cast<char*>(b), 2);
        header_len = b[0] | (b[1] << 8);
    } else {
        unsigned char b[4];
        in.read(reinterpret_cast<char*>(b), 4);
        header_len = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24);
    }
    std::string header(header_len, ' ');
    in.read(&header[0], header_len);
    if (header.find("'fortran_order': False") == std::string::npos) {
        *error = "only C-order arrays are supported"; return false;
    }
    const size_t d = header.find("'descr': '");
    if (d == std::string::npos) { *error = "no descr"; return false; }
    out->descr = header.substr(d + 10, header.find('\'', d + 10) - (d + 10));
    const size_t s = header.find("'shape': (");
    if (s == std::string::npos) { *error = "no shape"; return false; }
    std::istringstream shape(header.substr(s + 10));
    char comma;
    shape >> out->rows >> comma >> out->cols;
    if (!shape || out->rows <= 0 || out->cols <= 0) { *error = "need a 2-D shape"; return false; }
    const size_t item = (out->descr == "<f8") ? 8 : 1;
    if (out->descr != "<f8" && out->descr != "|u1" && out->descr != "|b1") {
        *error = "unsupported dtype " + out->descr; return false;
    }
    out->bytes.resize(static_cast<size_t>(out->rows * out->cols) * item);
    in.read(out->bytes.data(), static_cast<std::streamsize>(out->bytes.size()));
    if (!in) { *error = "truncated data"; return false; }
    return true;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 16) {
        std::fprintf(stderr, "usage: %s GRID.npy MASK.npy|- QUERIES.txt KIND A B C HAS_PEN I_MAX "
                             "PEN HAS_CAP CAP LEN_STRAIGHT LEN_DIAG REPEATS [DIST_OUT.bin]\n", argv[0]);
        return 2;
    }
    const auto t_load = std::chrono::steady_clock::now();
    Array grid, mask;
    std::string error;
    if (!read_npy(argv[1], &grid, &error) || grid.descr != "<f8") {
        std::fprintf(stderr, "grid: %s\n", error.empty() ? "need float64" : error.c_str());
        return 2;
    }
    const bool has_mask = std::string(argv[2]) != "-";
    if (has_mask && (!read_npy(argv[2], &mask, &error) || mask.rows != grid.rows
                     || mask.cols != grid.cols)) {
        std::fprintf(stderr, "mask: %s\n", error.empty() ? "shape differs" : error.c_str());
        return 2;
    }
    std::vector<std::pair<int64_t, int64_t>> queries;
    {
        std::ifstream q(argv[3]);
        int64_t s, t;
        while (q >> s >> t) queries.emplace_back(s, t);
    }
    const double load_ms = ms_since(t_load);
    const int kind = std::atoi(argv[4]);
    const double a = std::atof(argv[5]), b = std::atof(argv[6]), c = std::atof(argv[7]);
    const int has_pen = std::atoi(argv[8]);
    const double i_max = std::atof(argv[9]), pen = std::atof(argv[10]);
    const int has_cap = std::atoi(argv[11]);
    const double cap = std::atof(argv[12]);
    const double len_s = std::atof(argv[13]), len_d = std::atof(argv[14]);
    const int repeats = std::max(1, std::atoi(argv[15]));
    const char* dist_out = argc > 16 ? argv[16] : nullptr;

    const double* surf = reinterpret_cast<const double*>(grid.bytes.data());
    const uint8_t* m = has_mask ? reinterpret_cast<const uint8_t*>(mask.bytes.data()) : nullptr;
    const int64_t n = grid.rows * grid.cols;
    std::vector<double> dist(static_cast<size_t>(n));
    std::vector<uint8_t> dir(static_cast<size_t>(n));
    int64_t stats[4];

    std::printf("{\"load_ms\": %.6f, \"rows\": %lld, \"cols\": %lld, \"queries\": %zu}\n",
                load_ms, static_cast<long long>(grid.rows), static_cast<long long>(grid.cols),
                queries.size());
    for (size_t k = 0; k < queries.size(); ++k) {
        std::string times;
        int code = 0;
        for (int r = 0; r < repeats; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            code = gp_dijkstra(surf, m, grid.rows, grid.cols, 8, len_s, len_d, kind, a, b, c,
                               has_pen, i_max, pen, has_cap, cap, queries[k].first,
                               queries[k].second, 1, dist.data(), dir.data(), stats);
            char buf[32];
            std::snprintf(buf, sizeof buf, "%s%.6f", r ? ", " : "", ms_since(t0));
            times += buf;
            if (code != 0) break;
        }
        if (code != 0) {
            std::printf("{\"query\": %zu, \"error\": %d}\n", k, code);
            continue;
        }
        const double cost = dist[static_cast<size_t>(queries[k].second)];
        std::printf("{\"query\": %zu, \"start\": %lld, \"target\": %lld, \"cost\": %.17g, "
                    "\"expanded\": %lld, \"pushed\": %lld, \"reached\": %lld, "
                    "\"heap_bytes\": %lld, \"search_ms\": [%s]}\n",
                    k, static_cast<long long>(queries[k].first),
                    static_cast<long long>(queries[k].second),
                    std::isfinite(cost) ? cost : -1.0, static_cast<long long>(stats[0]),
                    static_cast<long long>(stats[1]), static_cast<long long>(stats[2]),
                    static_cast<long long>(stats[3]), times.c_str());
    }
    if (dist_out && !queries.empty()) {
        std::ofstream out(dist_out, std::ios::binary);
        out.write(reinterpret_cast<const char*>(dist.data()),
                  static_cast<std::streamsize>(dist.size() * sizeof(double)));
    }
    return 0;
}
