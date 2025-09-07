#ifndef DIGITS_GPU_HPP
#define DIGITS_GPU_HPP

#include "includes.hpp"
#include <mutex>
#include <unordered_set>
#include <condition_variable>
#include <atomic>
#include <vector>

#ifdef __CUDACC__
#	include <cuda_runtime.h>
#endif

struct GPUOperation {
    uint16_t flags;
    uint32_t result;
};

///-------- GPU functions --------///
// GPU API
bool gpu_combine_operations(const std::vector<operation>& op_map,
                            std::vector<operation>& new_ops,
                            size_t max_result);

// GPU port of build_operation_map
void build_operation_map_gpu(size_t max_ops, int max_concat_len);

// Declare globals from CPU world so CUDA can see them
extern std::vector<operation> operation_map;
extern bool VERBOSE;

// Locks and helpers needed by build_operation_map
static std::mutex op_map_mutex;
static std::mutex seen_mutex;
static std::mutex log_mutex;

static std::condition_variable cv_snapshot;
static std::atomic<size_t> last_saved_index;
static bool snapshot_stop_requested;

#endif // DIGITS_GPU_HPP
