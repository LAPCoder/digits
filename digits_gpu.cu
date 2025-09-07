#include "digits_gpu.hpp"
#include <vector>
#include <atomic>
#include <algorithm>
#include <iostream>

#ifdef __CUDACC__

inline uint64_t pack_key_(type result, numUsed flags)
{
	return (static_cast<uint64_t>(result) << 16) | static_cast<uint64_t>(flags);
}

__device__ uint64_t safe_pow_uint64_gpu(uint64_t base, uint64_t exp, uint64_t bound)
{
	uint64_t res = 1;
	while (exp)
	{
		if (exp & 1)
		{
			if (base != 0 && res > bound / base)
				return bound + 1;
			res *= base;
			if (res > bound)
				return bound + 1;
		}
		exp >>= 1;
		if (exp)
		{
			if (base != 0 && base > bound / base)
				base = bound + 1;
			else
				base *= base;
			if (base > bound)
				base = bound + 1;
		}
	}
	return res;
}

__global__ void combine_kernel(const GPUOperation *ops, size_t n,
							   GPUOperation *out_ops, unsigned long long *out_count,
							   uint32_t max_result)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t total = n * n;
	if (idx >= total)
		return;

	size_t i = idx / n;
	size_t j = idx % n;

	if (j < i)
		return; // lower triangle

	uint16_t flags_a = ops[i].flags;
	uint16_t flags_b = ops[j].flags;
	if ((flags_a & flags_b) != 0)
		return;

	uint16_t combined = flags_a | flags_b;
	uint32_t a = ops[i].result;
	uint32_t b = ops[j].result;

	// addition
	uint64_t sum = (uint64_t)a + (uint64_t)b;
	if (sum <= max_result)
	{
		unsigned long long pos = atomicAdd(out_count, 1ULL);
		out_ops[pos] = {combined, (uint32_t)sum};
	}

	// multiplication
	uint64_t prod = (uint64_t)a * (uint64_t)b;
	if (prod <= max_result)
	{
		unsigned long long pos = atomicAdd(out_count, 1ULL);
		out_ops[pos] = {combined, (uint32_t)prod};
	}

	// exponentiation a^b
	if (a >= 2 && b >= 1)
	{
		uint64_t pow_ab = safe_pow_uint64_gpu(a, b, max_result);
		if (pow_ab <= max_result)
		{
			unsigned long long pos = atomicAdd(out_count, 1ULL);
			out_ops[pos] = {combined, (uint32_t)pow_ab};
		}
	}

	// exponentiation b^a
	if (b >= 2 && a >= 1)
	{
		uint64_t pow_ba = safe_pow_uint64_gpu(b, a, max_result);
		if (pow_ba <= max_result)
		{
			unsigned long long pos = atomicAdd(out_count, 1ULL);
			out_ops[pos] = {combined, (uint32_t)pow_ba};
		}
	}
}

// Host wrapper
bool gpu_combine_operations(const std::vector<operation> &op_map,
							std::vector<operation> &new_ops,
							size_t max_result)
{
	if (op_map.empty())
		return true;

	size_t n = op_map.size();
	size_t total_pairs = n * n;

	// prepare GPUOperation array
	std::vector<GPUOperation> h_ops(n);
	for (size_t i = 0; i < n; ++i)
	{
		h_ops[i] = {op_map[i].flags, op_map[i].result};
	}

	GPUOperation *d_ops = nullptr;
	GPUOperation *d_out = nullptr;
	unsigned long long *d_count = nullptr;

	cudaMalloc(&d_ops, n * sizeof(GPUOperation));
	// allocate a generous buffer; host will trim results
	cudaMalloc(&d_out, total_pairs * 4 * sizeof(GPUOperation)); // rough upper bound
	cudaMalloc(&d_count, sizeof(unsigned long long));

	cudaMemcpy(d_ops, h_ops.data(), n * sizeof(GPUOperation), cudaMemcpyHostToDevice);
	cudaMemset(d_count, 0, sizeof(unsigned long long));

	size_t blockSize = 256;
	size_t gridSize = (total_pairs + blockSize - 1) / blockSize;

	combine_kernel<<<gridSize, blockSize>>>(d_ops, n, d_out, d_count, (uint32_t)max_result);
	cudaDeviceSynchronize();

	// copy count back
	unsigned long long out_count = 0;
	cudaMemcpy(&out_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	if (out_count == 0)
	{
		cudaFree(d_ops);
		cudaFree(d_out);
		cudaFree(d_count);
		return true;
	}

	std::vector<GPUOperation> h_out(out_count);
	cudaMemcpy(h_out.data(), d_out, out_count * sizeof(GPUOperation), cudaMemcpyDeviceToHost);

	// convert to std::vector<operation>
	new_ops.clear();
	new_ops.reserve(out_count);
	for (auto &g : h_out)
	{
		operation o;
		o.flags = g.flags;
		o.result = g.result;
		o.out = ""; // CPU will build string later
		new_ops.push_back(std::move(o));
	}

	cudaFree(d_ops);
	cudaFree(d_out);
	cudaFree(d_count);

	return true;
}

// GPU build_operation_map (multi-pass, hybride CPU/GPU)
void build_operation_map_gpu(size_t max_ops = MAX_PRECALC_OPS, int max_concat_len = 14) {
    // Phase 1: Concat (CPU)
    {
        std::lock_guard<std::mutex> lg(op_map_mutex);
        if (operation_map.capacity() < max_ops)
            operation_map.reserve(max_ops);
    }

    static std::unordered_set<uint64_t> global_seen;
    {
        std::lock_guard<std::mutex> lg(seen_mutex);
        global_seen.clear();
        global_seen.reserve(std::min<uint64_t>((uint64_t)max_ops * 2ULL, 100000000ULL));

        std::lock_guard<std::mutex> lg2(op_map_mutex);
        for (size_t i = 0; i < operation_map.size(); ++i) {
            global_seen.insert(pack_key_(operation_map[i].result, operation_map[i].flags));
        }
    }

    if (VERBOSE) {
        std::lock_guard<std::mutex> lg(log_mutex);
        std::clog << "[build][GPU] Starting concatenation (max_concat_len=" << max_concat_len << ")\n";
    }

    // DFS concaténation
    std::function<void(uint16_t, int, type, std::string &)> dfs_concat;
    dfs_concat = [&](uint16_t flags, int depth, type current_val, std::string &repr) {
        if (depth > max_concat_len) return;

        if (operation_map.size() >= max_ops) return;

        if (!repr.empty()) {
            if ((current_val > 0 && current_val <= MAX_RESULT) || (current_val == 0 && repr.size() == 1)) {
                uint64_t key = pack_key_(current_val, flags);
                bool inserted = false;
                {
                    std::lock_guard<std::mutex> lg(seen_mutex);
                    if (global_seen.insert(key).second)
                        inserted = true;
                }
                if (inserted) {
                    operation o{flags, current_val, repr};
                    std::lock_guard<std::mutex> lg(op_map_mutex);
                    if (operation_map.size() < max_ops)
                        operation_map.push_back(std::move(o));
                }
            }
        }
        if (depth == max_concat_len) return;

        for (int d = 0; d <= 9; ++d) {
            if (operation_map.size() >= max_ops) return;
            if (FLAG_OF_DIGIT(flags, d)) continue;

            if (repr.empty() && d == 0) {
                std::string tmp("0");
                uint16_t nf = flags | (1u << d);
                std::string saved = repr;
                repr = tmp;
                dfs_concat(nf, depth + 1, 0, repr);
                repr = saved;
                continue;
            }

            uint64_t nv = (uint64_t)current_val * 10ULL + (uint64_t)d;
            if (nv > MAX_RESULT) continue;

            std::string saved = repr;
            if (repr.empty())
                repr = std::to_string(d);
            else
                repr.push_back(char('0' + d));
            uint16_t nf = flags | (1u << d);
            dfs_concat(nf, depth + 1, (type)nv, repr);
            repr = saved;
        }
    };

    std::string rep;
    dfs_concat(0, 0, 0, rep);

    if (VERBOSE) {
        std::lock_guard<std::mutex> lg(log_mutex);
        std::clog << "[build][GPU] Concatenation finished ops=" << operation_map.size() << "\n";
    }

    // Phase 2: Combinaison (multi-pass GPU)
    size_t frontier_start = 0;
    const int MAX_GPU_PASSES = 50; // configurable

    for (int pass = 0; pass < MAX_GPU_PASSES; ++pass) {
        size_t frontier_size;
        {
            std::lock_guard<std::mutex> lg(op_map_mutex);
            frontier_size = operation_map.size() - frontier_start;
        }
        if (frontier_size == 0) {
            if (VERBOSE) {
                std::lock_guard<std::mutex> lg(log_mutex);
                std::clog << "[build][GPU] No new frontier, stopping\n";
            }
            break;
        }

        if (VERBOSE) {
            std::lock_guard<std::mutex> lg(log_mutex);
            std::clog << "[build][GPU] Pass " << pass
                      << " frontier=" << frontier_size
                      << " total_ops=" << operation_map.size() << "\n";
        }

        std::vector<operation> frontier;
        {
            std::lock_guard<std::mutex> lg(op_map_mutex);
            frontier.assign(operation_map.begin() + frontier_start, operation_map.end());
            frontier_start = operation_map.size();
        }

        std::vector<operation> new_ops;
        gpu_combine_operations(frontier, new_ops, MAX_RESULT);

        size_t inserted = 0;
        for (auto &op : new_ops) {
            if (operation_map.size() >= max_ops) break;

            uint64_t key = pack_key_(op.result, op.flags);
            bool unique = false;
            {
                std::lock_guard<std::mutex> lg(seen_mutex);
                if (global_seen.insert(key).second)
                    unique = true;
            }
            if (unique) {
                std::lock_guard<std::mutex> lg(op_map_mutex);
                if (operation_map.size() < max_ops) {
                    operation_map.push_back(op);
                    ++inserted;
                }
            }
        }

        if (VERBOSE) {
            std::lock_guard<std::mutex> lg(log_mutex);
            std::clog << "[build][GPU] GPU combine pass " << pass
                      << " inserted=" << inserted
                      << " total_ops=" << operation_map.size() << "\n";
        }

        if (inserted == 0) break; // stop si plus rien
    }
}

#endif // __CUDACC__
