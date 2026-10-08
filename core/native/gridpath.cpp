// Implicit-grid Dijkstra and height-area labelling for large terrains.
//
// Mirrors core/search.py exactly: the same neighbour order (core.grid.NEIGHBOR_OFFSETS),
// a min-heap on (distance, node) like heapq, strict relaxation, the same counters. The
// graph is never stored: neighbours and edge weights are computed when a node is expanded,
// so memory is O(cells), not O(edges). Built with -ffp-contract=off so a*b+c is never
// fused and the weights round exactly as numpy's do.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <queue>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define GP_EXPORT extern "C" __declspec(dllexport)
#else
#define GP_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

const int DI[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
const int DJ[8] = {0, -1, 1, 0, -1, 1, -1, 1};

enum CostKind { HEIGHT = 0, HEIGHT_TIEBREAK = 1, LENGTH_3D = 2, WEIGHTED = 3, LENGTH = 4 };

struct Cost {
    int kind;
    double a, b, c;        // weights of |dh|, length, hypot(dh, length); b is the tie-break
    int has_penalty;
    double i_max, penalty; // gradient_penalised(base, i_max_percent, penalty_per_percent)
    int has_cap;
    double cap_ratio;      // max |dh| / length, i.e. max_abs_gradient_percent / 100

    // true when the edge exists; the weight is written to *w
    inline bool weight(double dh, double len, double* w) const {
        const double adh = std::fabs(dh);
        if (has_cap && !(adh <= cap_ratio * len)) return false;
        double out;
        switch (kind) {
            case HEIGHT: out = adh; break;
            case HEIGHT_TIEBREAK: out = adh + b * len; break;
            case LENGTH_3D: out = std::hypot(dh, len); break;
            case LENGTH: out = len; break;
            default:
                out = 0.0;
                if (a != 0.0) out += a * adh;
                if (b != 0.0) out += b * len;
                if (c != 0.0) out += c * std::hypot(dh, len);
        }
        if (has_penalty) {
            const double gradient = 100.0 * adh / len;
            double excess = gradient - i_max;
            if (excess < 0.0) excess = 0.0;
            out = out + (penalty * excess) * len;
        }
        *w = out;
        return true;
    }
};

typedef std::pair<double, int64_t> Item;

}  // namespace

GP_EXPORT int gp_abi_version() { return 1; }

// Directed edge count of the (masked, capped) grid graph, like GridGraph.n_edges.
GP_EXPORT int64_t gp_count_edges(const double* surf, const uint8_t* mask, int64_t n_rows,
                                 int64_t n_cols, int connectivity, double len_straight,
                                 double len_diag, int has_cap, double cap_ratio) {
    int64_t count = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            const int64_t u = i * n_cols + j;
            if (mask && !mask[u]) continue;
            for (int k = 0; k < connectivity; ++k) {
                const int64_t i2 = i + DI[k], j2 = j + DJ[k];
                if (i2 < 0 || i2 >= n_rows || j2 < 0 || j2 >= n_cols) continue;
                const int64_t v = i2 * n_cols + j2;
                if (mask && !mask[v]) continue;
                const double len = k < 4 ? len_straight : len_diag;
                if (has_cap && !(std::fabs(surf[v] - surf[u]) <= cap_ratio * len)) continue;
                ++count;
            }
        }
    }
    return count;
}

// Returns 0 on success, 1 on allocation failure, 2 on bad arguments.
// out_dist: n doubles (inf where unreached). out_dir: n bytes, the offset index k in
// 0..7 of the step that reached the node from its predecessor, 255 where none.
// stats: [nodes_expanded, nodes_pushed, nodes_reached, peak_heap_bytes].
GP_EXPORT int gp_dijkstra(const double* surf, const uint8_t* mask, int64_t n_rows,
                          int64_t n_cols, int connectivity, double len_straight,
                          double len_diag, int kind, double a, double b, double c,
                          int has_penalty, double i_max, double penalty, int has_cap,
                          double cap_ratio, int64_t start, int64_t target,
                          int stop_at_target, double* out_dist, uint8_t* out_dir,
                          int64_t* stats) {
    const int64_t n = n_rows * n_cols;
    if (start < 0 || start >= n || target < 0 || target >= n) return 2;
    if (connectivity != 4 && connectivity != 8) return 2;
    const Cost cost = {kind, a, b, c, has_penalty, i_max, penalty, has_cap, cap_ratio};
    try {
        std::vector<uint8_t> settled(static_cast<size_t>(n), 0);
        const double inf = std::numeric_limits<double>::infinity();
        for (int64_t k = 0; k < n; ++k) out_dist[k] = inf;
        std::memset(out_dir, 255, static_cast<size_t>(n));
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
        out_dist[start] = 0.0;
        heap.push(Item(0.0, start));
        int64_t expanded = 0, pushed = 1;
        size_t peak = 1;
        while (!heap.empty()) {
            const Item top = heap.top();
            heap.pop();
            const int64_t node = top.second;
            if (settled[node]) continue;
            settled[node] = 1;
            ++expanded;
            if (stop_at_target && node == target) break;
            if (mask && !mask[node]) continue;
            const int64_t i = node / n_cols, j = node % n_cols;
            const double h = surf[node];
            for (int k = 0; k < connectivity; ++k) {
                const int64_t i2 = i + DI[k], j2 = j + DJ[k];
                if (i2 < 0 || i2 >= n_rows || j2 < 0 || j2 >= n_cols) continue;
                const int64_t v = i2 * n_cols + j2;
                if (mask && !mask[v]) continue;
                double w;
                if (!cost.weight(surf[v] - h, k < 4 ? len_straight : len_diag, &w)) continue;
                const double candidate = top.first + w;
                if (candidate < out_dist[v]) {
                    out_dist[v] = candidate;
                    out_dir[v] = static_cast<uint8_t>(k);
                    heap.push(Item(candidate, v));
                    ++pushed;
                    if (heap.size() > peak) peak = heap.size();
                }
            }
        }
        int64_t reached = 0;
        for (int64_t k = 0; k < n; ++k) reached += out_dist[k] < inf;
        stats[0] = expanded;
        stats[1] = pushed;
        stats[2] = reached;
        stats[3] = static_cast<int64_t>(peak * sizeof(Item) + static_cast<size_t>(n));
    } catch (const std::bad_alloc&) {
        return 1;
    }
    return 0;
}

// Connected components of equal `bucket` values (bucket < 0 = nodata, label -1).
// Labels are numbered like core.hag.build_hag: by bucket ascending, then by the raster
// position of each component's first cell, which is the order scipy.ndimage.label uses.
// Returns the number of areas, or -1 on allocation failure.
GP_EXPORT int64_t gp_label_areas(const int32_t* bucket, int64_t n_rows, int64_t n_cols,
                                 int connectivity, int64_t* out_labels) {
    const int64_t n = n_rows * n_cols;
    try {
        for (int64_t k = 0; k < n; ++k) out_labels[k] = -1;
        std::vector<int32_t> comp_bucket;
        std::vector<int64_t> stack;
        int64_t n_comp = 0;
        for (int64_t seed = 0; seed < n; ++seed) {
            if (bucket[seed] < 0 || out_labels[seed] >= 0) continue;
            const int32_t value = bucket[seed];
            out_labels[seed] = n_comp;
            stack.clear();
            stack.push_back(seed);
            while (!stack.empty()) {
                const int64_t node = stack.back();
                stack.pop_back();
                const int64_t i = node / n_cols, j = node % n_cols;
                for (int k = 0; k < connectivity; ++k) {
                    const int64_t i2 = i + DI[k], j2 = j + DJ[k];
                    if (i2 < 0 || i2 >= n_rows || j2 < 0 || j2 >= n_cols) continue;
                    const int64_t v = i2 * n_cols + j2;
                    if (out_labels[v] >= 0 || bucket[v] != value) continue;
                    out_labels[v] = n_comp;
                    stack.push_back(v);
                }
            }
            comp_bucket.push_back(value);
            ++n_comp;
        }
        // components are already in raster order of their first cell; a stable counting
        // sort by bucket gives (bucket, first cell) order
        int32_t lo = 0, hi = -1;
        for (int64_t k = 0; k < n_comp; ++k) {
            if (k == 0 || comp_bucket[k] < lo) lo = comp_bucket[k];
            if (k == 0 || comp_bucket[k] > hi) hi = comp_bucket[k];
        }
        std::vector<int64_t> offset(static_cast<size_t>(hi - lo + 2), 0);
        for (int64_t k = 0; k < n_comp; ++k) ++offset[comp_bucket[k] - lo + 1];
        for (size_t k = 1; k < offset.size(); ++k) offset[k] += offset[k - 1];
        std::vector<int64_t> renumber(static_cast<size_t>(n_comp));
        for (int64_t k = 0; k < n_comp; ++k) renumber[k] = offset[comp_bucket[k] - lo]++;
        for (int64_t k = 0; k < n; ++k)
            if (out_labels[k] >= 0) out_labels[k] = renumber[out_labels[k]];
        return n_comp;
    } catch (const std::bad_alloc&) {
        return -1;
    }
}
