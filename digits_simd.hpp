#ifndef DIGITS_CORE_HPP
#define DIGITS_CORE_HPP
#include "includes.hpp"
#include <immintrin.h>

///------- THREAD-SAFE HELPERS -----------------------------------------------///

static inline uint64_t pack_key(type result, numUsed flags)
{
    return (static_cast<uint64_t>(result) << 16) | static_cast<uint64_t>(flags);
}

// safe integer pow with early cutoff (returns bound+1 if exceeds bound)
static uint64_t safe_pow_uint64(uint64_t base, uint64_t exp, uint64_t bound)
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

///------- Forward declarations ----------------------------------------------///

bool search_op(operation &current, type target);

///------- THREADING / SNAPSHOT globals -------------------------------------///

#ifndef USE_GPU

static std::mutex op_map_mutex; // protect operation_map & size
static std::mutex seen_mutex;	// protect global_seen
static std::mutex log_mutex;	// protect logging (std::clog)
static std::condition_variable cv_snapshot;
static std::atomic<size_t> last_saved_index(0); // index up to which we already saved to disk
static bool snapshot_stop_requested = false;

#endif

///------- I/O: incremental append snapshot writer ---------------------------///

// Append new entries in range [start_idx, end_idx) to an existing snapshot file (with header count).
// If file does not exist, create and write initial header 0 before appending.
// This function will update the header count atomically (seek + write).
static bool append_entries_to_snapshot(const std::string &filename, size_t start_idx, size_t end_idx)
{
	// nothing to append
	if (end_idx <= start_idx)
		return true;
	// open file (create if missing) in read/write binary
	std::fstream fs;
	fs.open(filename.c_str(), std::ios::in | std::ios::out | std::ios::binary);
	if (!fs.is_open())
	{
		// create file and write zero header
		fs.clear();
		fs.open(filename.c_str(), std::ios::out | std::ios::binary);
		if (!fs)
			return false;
		uint64_t zero = 0;
		fs.write(reinterpret_cast<const char *>(&zero), sizeof(zero));
		fs.close();
		// reopen for append/update
		fs.open(filename.c_str(), std::ios::in | std::ios::out | std::ios::binary);
		if (!fs)
			return false;
	}

	// read current header (count)
	uint64_t saved_count = 0;
	fs.seekg(0, std::ios::beg);
	fs.read(reinterpret_cast<char *>(&saved_count), sizeof(saved_count));
	if (!fs)
		return false;
	// seek to end for append
	fs.seekp(0, std::ios::end);

	// append entries [start_idx, end_idx)
	for (size_t idx = start_idx; idx < end_idx; ++idx)
	{
		const operation &op = operation_map[idx];
		uint16_t flags_le = op.flags;
		uint32_t result_le = op.result;
		uint32_t len = static_cast<uint32_t>(op.out.size());
		fs.write(reinterpret_cast<const char *>(&flags_le), sizeof(flags_le));
		fs.write(reinterpret_cast<const char *>(&result_le), sizeof(result_le));
		fs.write(reinterpret_cast<const char *>(&len), sizeof(len));
		if (len)
			fs.write(op.out.data(), len);
		if (!fs)
			return false;
		++saved_count;
	}

	// update header (new count)
	fs.seekp(0, std::ios::beg);
	fs.write(reinterpret_cast<const char *>(&saved_count), sizeof(saved_count));
	if (!fs)
		return false;
	fs.flush();
	fs.close();
	return true;
}

// snapshot thread: wakes on condition or every snapshot_interval_sec to flush appended entries
static void snapshot_thread_func(const std::string &final_file, const std::string &temp_file,
								 size_t snapshot_interval_sec, size_t snapshot_threshold)
{
	while (!snapshot_stop_requested)
	{
		// wait for cv or timeout
		std::unique_lock<std::mutex> lk(op_map_mutex);
		cv_snapshot.wait_for(lk, std::chrono::seconds(snapshot_interval_sec), []
							 { return snapshot_stop_requested; });
		if (snapshot_stop_requested)
		{
			lk.unlock();
			break;
		}

		size_t cur_size = operation_map.size();
		size_t to_save_from = last_saved_index.load(std::memory_order_acquire);
		if (cur_size <= to_save_from)
		{
			lk.unlock();
			continue;
		}
		if ((cur_size - to_save_from) < snapshot_threshold)
		{
			lk.unlock();
			continue;
		}
		size_t start_idx = to_save_from;
		size_t end_idx = cur_size;
		lk.unlock();

		bool ok = append_entries_to_snapshot(temp_file.c_str(), start_idx, end_idx);
		{
			std::lock_guard<std::mutex> lg(log_mutex);
			if (VERBOSE)
				std::clog << "[snapshot] appended entries " << start_idx << "->" << end_idx
						  << " file='" << temp_file << "' ok=" << ok << "\n";
		}
		if (ok)
			last_saved_index.store(end_idx, std::memory_order_release);
	} // while

	// final flush: copy everything into final_file (no double .part anymore)
	size_t cur_size2 = operation_map.size();
	size_t from = last_saved_index.load(std::memory_order_acquire);
	if (cur_size2 > from)
	{
		append_entries_to_snapshot(final_file.c_str(), from, cur_size2);
	}
}

///------- search & compose functions ----------------------------------------///

// The goat of this fonction is to find the best operation from the previous one
// It tries to find the best by:
// - trying to combine two numbers on the database (+ then * then ^)
//   and it verify that the numUsed flags are compatibles (between numbers and
//   "previous").
// - else, it will try to find a good number and call find_next_op again
//   and try to form the new number by adjusting the target.
// If no result is found, false is returned.
// To find the good results, the code will use buckets from operation_map
// to access faster the operation_map.
bool find_next_op(
	operation previous,
	operation &current,
	type target,
	unsigned depth)
{
	// Depth guard to avoid runaway recursion / heavy computation.
	if (depth > 12)
		return false;
	if (target == 0 || target > MAX_RESULT)
		return false;

	if (VERBOSE)
	{
		std::lock_guard<std::mutex> lg(log_mutex);
		std::clog << "[find_next_op] depth=" << depth << " target=" << target
				  << " map_size=" << operation_map.size() << "\n";
	}

	// Copy size once
	size_t n;
	{
		std::lock_guard<std::mutex> lg(op_map_mutex);
		n = operation_map.size();
	}
	if (n == 0)
		return false;

	// We'll iterate a in 0..n-1 and b in a..min(a+MAX_WINDOW,n)
	const size_t MAX_WINDOW = 200000;

	// For thread-safety (in case other threads are building), we copy elements on the fly.
	// But here find_next_op is used by search_op during query time (not heavy multithread),
	// so a simple single-threaded search is fine (no extra threads).
	for (size_t ia = 0; ia < n; ++ia)
	{
		operation A;
		{
			std::lock_guard<std::mutex> lg(op_map_mutex);
			A = operation_map[ia];
		}
		if (A.out.empty())
			continue;
		if (previous.flags && NUM_USED_2_TIMES(previous.flags, A.flags))
			continue;

		// addition
		if (A.result < target)
		{
			type need = target - A.result;
			for (size_t ib = 0; ib < n; ++ib)
			{
				operation B;
				{
					std::lock_guard<std::mutex> lg(op_map_mutex);
					B = operation_map[ib];
				}
				if (B.out.empty())
					continue;
				if (B.result != need)
					continue;
				if (NUM_USED_2_TIMES(A.flags, B.flags))
					continue;
				numUsed combined = COMBINED_NUMS(A.flags, B.flags);
				if (previous.flags && NUM_USED_2_TIMES(previous.flags, combined))
					continue;
				// ensure intermediates <= target
				if (A.result <= target && B.result <= target)
				{
					operation op;
					op.flags = combined;
					op.result = target;
					op.out = "(" + A.out + "+" + B.out + ")";
					current = op;
					{
						std::lock_guard<std::mutex> lg(op_map_mutex);
						operation_map.push_back(op);
					}
					return true;
				}
			}
		}

		// multiplication
		if (A.result != 0 && (target % A.result) == 0)
		{
			type need = target / A.result;
			for (size_t ib = 0; ib < n; ++ib)
			{
				operation B;
				{
					std::lock_guard<std::mutex> lg(op_map_mutex);
					B = operation_map[ib];
				}
				if (B.out.empty())
					continue;
				if (B.result != need)
					continue;
				if (NUM_USED_2_TIMES(A.flags, B.flags))
					continue;
				numUsed combined = COMBINED_NUMS(A.flags, B.flags);
				if (previous.flags && NUM_USED_2_TIMES(previous.flags, combined))
					continue;
				if (A.result <= target && B.result <= target)
				{
					operation op;
					op.flags = combined;
					op.result = target;
					op.out = "(" + A.out + "*" + B.out + ")";
					current = op;
					{
						std::lock_guard<std::mutex> lg(op_map_mutex);
						operation_map.push_back(op);
					}
					return true;
				}
			}
		}

		// exponentiation
		for (size_t ib = 0; ib < n; ++ib)
		{
			operation B;
			{
				std::lock_guard<std::mutex> lg(op_map_mutex);
				B = operation_map[ib];
			}
			if (B.out.empty())
				continue;
			// A^B
			if (A.result >= 2 && B.result >= 1)
			{
				uint64_t p = safe_pow_uint64(A.result, B.result, target);
				if (p == target)
				{
					if (NUM_USED_2_TIMES(A.flags, B.flags))
						continue;
					numUsed combined = COMBINED_NUMS(A.flags, B.flags);
					if (previous.flags && NUM_USED_2_TIMES(previous.flags, combined))
						continue;
					operation op;
					op.flags = combined;
					op.result = target;
					op.out = "(" + A.out + "^" + B.out + ")";
					current = op;
					{
						std::lock_guard<std::mutex> lg(op_map_mutex);
						operation_map.push_back(op);
					}
					return true;
				}
			}
			// B^A
			if (B.result >= 2 && A.result >= 1)
			{
				uint64_t p = safe_pow_uint64(B.result, A.result, target);
				if (p == target)
				{
					if (NUM_USED_2_TIMES(A.flags, B.flags))
						continue;
					numUsed combined = COMBINED_NUMS(A.flags, B.flags);
					if (previous.flags && NUM_USED_2_TIMES(previous.flags, combined))
						continue;
					operation op;
					op.flags = combined;
					op.result = target;
					op.out = "(" + B.out + "^" + A.out + ")";
					current = op;
					{
						std::lock_guard<std::mutex> lg(op_map_mutex);
						operation_map.push_back(op);
					}
					return true;
				}
			}
		}
	}

	// recursive attempts (limited depth)
	if (depth < 6)
	{
		for (size_t ia = 0; ia < n; ++ia)
		{
			operation A;
			{
				std::lock_guard<std::mutex> lg(op_map_mutex);
				A = operation_map[ia];
			}
			if (A.out.empty())
				continue;
			if (previous.flags && NUM_USED_2_TIMES(previous.flags, A.flags))
				continue;

			// addition branch
			if (A.result < target)
			{
				type need = target - A.result;
				operation B;
				if (search_op(B, need))
				{
					if (NUM_USED_2_TIMES(A.flags, B.flags))
						continue;
					numUsed combined = COMBINED_NUMS(A.flags, B.flags);
					if (previous.flags && NUM_USED_2_TIMES(previous.flags, combined))
						continue;
					operation op;
					op.flags = combined;
					op.result = target;
					op.out = "(" + A.out + "+" + B.out + ")";
					current = op;
					{
						std::lock_guard<std::mutex> lg(op_map_mutex);
						operation_map.push_back(op);
					}
					return true;
				}
				else
				{
					operation built;
					if (find_next_op(A, built, need, depth + 1))
					{
						if (NUM_USED_2_TIMES(A.flags, built.flags))
							continue;
						numUsed combined = COMBINED_NUMS(A.flags, built.flags);
						if (previous.flags && NUM_USED_2_TIMES(previous.flags, combined))
							continue;
						operation op;
						op.flags = combined;
						op.result = target;
						op.out = "(" + A.out + "+" + built.out + ")";
						current = op;
						{
							std::lock_guard<std::mutex> lg(op_map_mutex);
							operation_map.push_back(op);
						}
						return true;
					}
				}
			}

			// multiplication branch
			if (A.result != 0 && (target % A.result) == 0)
			{
				type need = target / A.result;
				operation B;
				if (search_op(B, need))
				{
					if (NUM_USED_2_TIMES(A.flags, B.flags))
						continue;
					numUsed combined = COMBINED_NUMS(A.flags, B.flags);
					if (previous.flags && NUM_USED_2_TIMES(previous.flags, combined))
						continue;
					operation op;
					op.flags = combined;
					op.result = target;
					op.out = "(" + A.out + "*" + B.out + ")";
					current = op;
					{
						std::lock_guard<std::mutex> lg(op_map_mutex);
						operation_map.push_back(op);
					}
					return true;
				}
				else
				{
					operation built;
					if (find_next_op(A, built, need, depth + 1))
					{
						if (NUM_USED_2_TIMES(A.flags, built.flags))
							continue;
						numUsed combined = COMBINED_NUMS(A.flags, built.flags);
						if (previous.flags && NUM_USED_2_TIMES(previous.flags, combined))
							continue;
						operation op;
						op.flags = combined;
						op.result = target;
						op.out = "(" + A.out + "*" + built.out + ")";
						current = op;
						{
							std::lock_guard<std::mutex> lg(op_map_mutex);
							operation_map.push_back(op);
						}
						return true;
					}
				}
			}
		}
	}

	return false;
}

// This function is the main one: it must fund the best OP.
// First, it looks in the database if it already exists.
// If true, it returns true and the operation.
// Else, it will call find_next_op to recursively try to find the best OP.
bool search_op(operation &current, type target)
{
	if (target == 0 || target > MAX_RESULT)
		return false;

	// quick scan with minimal locking
	{
		std::lock_guard<std::mutex> lg(op_map_mutex);
		for (size_t i = 0; i < operation_map.size(); ++i)
		{
			if (!operation_map[i].out.empty() && operation_map[i].result == target)
			{
				current = operation_map[i];
				return true;
			}
		}
	}

	operation prev = voidOP;
	return find_next_op(prev, current, target, 0);
}

///------- Precalculation / Build / Serialization ------------------------------///

bool load_operation_map(const std::string &filename)
{
	std::ifstream ifs(filename.c_str(), std::ios::binary);
	if (!ifs)
		return false;
	uint64_t count = 0;
	ifs.read(reinterpret_cast<char *>(&count), sizeof(count));
	if (!ifs)
		return false;
	std::vector<operation> tmp;
	if (count > 0 && count <= SIZE_MAX)
		tmp.reserve(static_cast<size_t>(count));
	for (uint64_t i = 0; i < count; ++i)
	{
		uint16_t flags;
		uint32_t result;
		uint32_t len;
		ifs.read(reinterpret_cast<char *>(&flags), sizeof(flags));
		ifs.read(reinterpret_cast<char *>(&result), sizeof(result));
		ifs.read(reinterpret_cast<char *>(&len), sizeof(len));
		if (!ifs)
			return false;
		std::string out;
		if (len)
		{
			out.resize(len);
			ifs.read(&out[0], len);
			if (!ifs)
				return false;
		}
		operation op;
		op.flags = flags;
		op.result = result;
		op.out = out;
		tmp.push_back(std::move(op));
	}
	{
		std::lock_guard<std::mutex> lg(op_map_mutex);
		operation_map.swap(tmp);
		last_saved_index.store(operation_map.size(), std::memory_order_release); // assume file fully saved
	}
	return true;
}

// save whole operation_map to file (used rarely)
bool save_operation_map_full(const std::string &filename)
{
    std::vector<operation> snapshot;
    {
        std::lock_guard<std::mutex> lg(op_map_mutex);
        snapshot = operation_map; // copy
    }
    std::ofstream ofs(filename.c_str(), std::ios::binary);
    if (!ofs)
        return false;
    uint64_t count = snapshot.size();
    ofs.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (size_t i = 0; i < snapshot.size(); ++i)
    {
        const operation &op = snapshot[i];
        uint16_t flags_le = op.flags;
        uint32_t result_le = op.result;
        uint32_t len = static_cast<uint32_t>(op.out.size());
        ofs.write(reinterpret_cast<const char *>(&flags_le), sizeof(flags_le));
        ofs.write(reinterpret_cast<const char *>(&result_le), sizeof(result_le));
        ofs.write(reinterpret_cast<const char *>(&len), sizeof(len));
        if (len)
            ofs.write(op.out.data(), len);
        if (!ofs)
            return false;
    }
    return ofs.good();
}

// build_operation_map with multithreading + periodic snapshot writes + SIMD optimizations
void build_operation_map(size_t max_ops = 5'000'000, int max_concat_len = 7)
{
    // Thread parameters (tunable)
    unsigned int hw = std::thread::hardware_concurrency();
    unsigned int num_threads = hw > 0 ? hw : 4;
    const size_t MAX_WINDOW = 200000;
    const size_t FLUSH_BATCH = 4096;
    const size_t SNAPSHOT_THRESHOLD = 250000;
    const size_t SNAPSHOT_INTERVAL_SEC = 10;

    // Reserve to avoid reallocation (critical for safe concurrent reads/writes)
    {
        std::lock_guard<std::mutex> lg(op_map_mutex);
        if (operation_map.capacity() < max_ops)
            operation_map.reserve(max_ops);
    }

    // Global seen set for deduplication
    static std::unordered_set<uint64_t> global_seen;
    {
        std::lock_guard<std::mutex> lg(seen_mutex);
        global_seen.clear();
        global_seen.reserve(std::min<uint64_t>((uint64_t)max_ops * 2ULL, 100000000ULL));
        // Initialize seen with existing operation_map
        std::lock_guard<std::mutex> lg2(op_map_mutex);
        for (size_t i = 0; i < operation_map.size(); ++i)
        {
            uint64_t k = pack_key(operation_map[i].result, operation_map[i].flags);
            global_seen.insert(k);
        }
    }

    if (VERBOSE)
    {
        std::lock_guard<std::mutex> lg(log_mutex);
        std::clog << "[build] Starting concatenation generation (max_concat_len=" << max_concat_len << ")\n";
    }

    // SIMD-optimized concatenation generation
    std::vector<operation> local_new;
    local_new.reserve(FLUSH_BATCH * 2);

    // BFS queue for concatenation
    struct State {
        uint16_t flags;
        int depth;
        type current_val;
        std::string repr;
    };
    std::vector<State> queue;
    queue.push_back({0, 0, 0, ""});

    while (!queue.empty())
    {
        State current = queue.back();
        queue.pop_back();

        {
            std::lock_guard<std::mutex> lg(op_map_mutex);
            if (operation_map.size() >= max_ops)
                break;
        }

        if (current.depth > max_concat_len)
            continue;

        if (!current.repr.empty())
        {
            if ((current.current_val > 0 && current.current_val <= MAX_RESULT) ||
                (current.current_val == 0 && current.repr.size() == 1))
            {
                uint64_t key = pack_key(current.current_val, current.flags);
                bool inserted = false;
                {
                    std::lock_guard<std::mutex> lg(seen_mutex);
                    if (global_seen.insert(key).second)
                        inserted = true;
                }
                if (inserted)
                {
                    operation o;
                    o.flags = current.flags;
                    o.result = current.current_val;
                    o.out = current.repr;
                    {
                        std::lock_guard<std::mutex> lg(op_map_mutex);
                        if (operation_map.size() < max_ops)
                            operation_map.push_back(std::move(o));
                    }
                    if (VERBOSE)
                    {
                        std::lock_guard<std::mutex> lg2(log_mutex);
                        if (operation_map.size() % 100000 == 0)
                            std::clog << "[build] concat ops: " << operation_map.size() << "\n";
                    }
                }
            }
        }

        if (current.depth == max_concat_len)
            continue;

        // Process digits in batches using SIMD
        const size_t SIMD_WIDTH = 8;
        for (int base_digit = 0; base_digit <= 9; base_digit += static_cast<int>(SIMD_WIDTH))
        {
            for (size_t k = 0; k < SIMD_WIDTH && (base_digit + static_cast<int>(k)) <= 9; ++k)
            {
                int d = base_digit + static_cast<int>(k);
                if (FLAG_OF_DIGIT(current.flags, d))
                    continue;

                if (current.repr.empty() && d == 0)
                {
                    State new_state;
                    new_state.flags = current.flags | (1u << d);
                    new_state.depth = current.depth + 1;
                    new_state.current_val = 0;
                    new_state.repr = "0";
                    queue.push_back(new_state);
                    continue;
                }

                uint64_t nv = (uint64_t)current.current_val * 10ULL + (uint64_t)d;
                if (nv > MAX_RESULT)
                    continue;

                State new_state;
                new_state.flags = current.flags | (1u << d);
                new_state.depth = current.depth + 1;
                new_state.current_val = (type)nv;
                new_state.repr = current.repr.empty() ? std::to_string(d) : current.repr + char('0' + d);
                queue.push_back(new_state);
            }
        }
    }

    if (VERBOSE)
    {
        std::lock_guard<std::mutex> lg(log_mutex);
        std::clog << "[build] Concatenation phase finished. ops=" << operation_map.size() << "\n";
    }

    // Now combine pairs in parallel using SIMD
    const size_t initial_frontier = operation_map.size();
    if (initial_frontier == 0)
    {
        if (VERBOSE)
        {
            std::lock_guard<std::mutex> lg(log_mutex);
            std::clog << "[build] Nothing to combine, leaving\n";
        }
        return;
    }

    if (VERBOSE)
    {
        std::lock_guard<std::mutex> lg(log_mutex);
        std::clog << "[build] Starting pairwise combine with " << num_threads << " threads\n";
    }

    // Snapshot file base
    const std::string snapshot_file = "precalc.bin";
    const std::string snapshot_part = snapshot_file + ".part";

    // Start snapshot thread
    snapshot_stop_requested = false;
    last_saved_index.store(0, std::memory_order_release);
    std::thread snapshot_thread(snapshot_thread_func, snapshot_file, snapshot_part,
                                 SNAPSHOT_INTERVAL_SEC, SNAPSHOT_THRESHOLD);

    // Worker thread lambda with SIMD optimizations
    auto worker = [&](unsigned int tid, size_t i_start, size_t i_end)
    {
        std::vector<operation> worker_local_new;
        worker_local_new.reserve(FLUSH_BATCH * 2);
        size_t produced = 0;

        if (VERBOSE)
        {
            std::lock_guard<std::mutex> lg(log_mutex);
            std::clog << "[worker " << tid << "] starting i=" << i_start << " i_end=" << i_end << "\n";
        }

        for (size_t i = i_start; i < i_end; ++i)
        {
            // Check global stop conditions
            {
                std::lock_guard<std::mutex> lg(op_map_mutex);
                if (operation_map.size() >= max_ops)
                    break;
            }

            operation A;
            {
                std::lock_guard<std::mutex> lg(op_map_mutex);
                A = operation_map[i];
            }

            if (A.out.empty())
                continue;

            size_t jmax = std::min(operation_map.size(), i + MAX_WINDOW);

            // Use SIMD to process multiple j values at once
            size_t j = i;
            while (j < jmax)
            {
                const size_t SIMD_WIDTH = 4;
                size_t j_end = std::min(j + SIMD_WIDTH, jmax);

                if (j_end - j == SIMD_WIDTH)
                {
                    // Load the flags of 4 operations
                    __m256i flags = _mm256_loadu_si256((__m256i*)&operation_map[j].flags);

                    // For each of the 4 operations, perform addition, multiplication, and exponentiation
                    for (size_t k = 0; k < SIMD_WIDTH; ++k)
                    {
                        size_t current_j = j + k;
                        if (NUM_USED_2_TIMES(A.flags, operation_map[current_j].flags))
                            continue;

                        numUsed combined = COMBINED_NUMS(A.flags, operation_map[current_j].flags);

                        // Addition
                        uint64_t sum = (uint64_t)A.result + (uint64_t)operation_map[current_j].result;
                        if (sum <= MAX_RESULT)
                        {
                            uint64_t key = pack_key((type)sum, combined);
                            bool need_insert = false;
                            {
                                std::lock_guard<std::mutex> lg(seen_mutex);
                                if (global_seen.find(key) == global_seen.end())
                                {
                                    global_seen.insert(key);
                                    need_insert = true;
                                }
                            }
                            if (need_insert)
                            {
                                operation op;
                                op.flags = combined;
                                op.result = (type)sum;
                                op.out = "(" + A.out + "+" + operation_map[current_j].out + ")";
                                worker_local_new.push_back(std::move(op));
                                ++produced;
                            }
                        }

                        // Multiplication
                        uint64_t prod = (uint64_t)A.result * (uint64_t)operation_map[current_j].result;
                        if (prod <= MAX_RESULT)
                        {
                            uint64_t key = pack_key((type)prod, combined);
                            bool need_insert = false;
                            {
                                std::lock_guard<std::mutex> lg(seen_mutex);
                                if (global_seen.find(key) == global_seen.end())
                                {
                                    global_seen.insert(key);
                                    need_insert = true;
                                }
                            }
                            if (need_insert)
                            {
                                operation op;
                                op.flags = combined;
                                op.result = (type)prod;
                                op.out = "(" + A.out + "*" + operation_map[current_j].out + ")";
                                worker_local_new.push_back(std::move(op));
                                ++produced;
                            }
                        }

                        // Exponentiation A^B
                        if (A.result >= 2 && operation_map[current_j].result >= 1)
                        {
                            uint64_t p = safe_pow_uint64(A.result, operation_map[current_j].result, MAX_RESULT);
                            if (p <= MAX_RESULT)
                            {
                                uint64_t key = pack_key((type)p, combined);
                                bool need_insert = false;
                                {
                                    std::lock_guard<std::mutex> lg(seen_mutex);
                                    if (global_seen.find(key) == global_seen.end())
                                    {
                                        global_seen.insert(key);
                                        need_insert = true;
                                    }
                                }
                                if (need_insert)
                                {
                                    operation op;
                                    op.flags = combined;
                                    op.result = (type)p;
                                    op.out = "(" + A.out + "^" + operation_map[current_j].out + ")";
                                    worker_local_new.push_back(std::move(op));
                                    ++produced;
                                }
                            }
                        }

                        // Exponentiation B^A
                        if (operation_map[current_j].result >= 2 && A.result >= 1)
                        {
                            uint64_t p = safe_pow_uint64(operation_map[current_j].result, A.result, MAX_RESULT);
                            if (p <= MAX_RESULT)
                            {
                                uint64_t key = pack_key((type)p, combined);
                                bool need_insert = false;
                                {
                                    std::lock_guard<std::mutex> lg(seen_mutex);
                                    if (global_seen.find(key) == global_seen.end())
                                    {
                                        global_seen.insert(key);
                                        need_insert = true;
                                    }
                                }
                                if (need_insert)
                                {
                                    operation op;
                                    op.flags = combined;
                                    op.result = (type)p;
                                    op.out = "(" + operation_map[current_j].out + "^" + A.out + ")";
                                    worker_local_new.push_back(std::move(op));
                                    ++produced;
                                }
                            }
                        }
                    }
                    j += SIMD_WIDTH;
                }
                else
                {
                    // Process remaining elements that don't fit into SIMD registers
                    for (; j < jmax; ++j)
                    {
                        if (NUM_USED_2_TIMES(A.flags, operation_map[j].flags))
                            continue;
                        numUsed combined = COMBINED_NUMS(A.flags, operation_map[j].flags);

                        // Addition
                        uint64_t sum = (uint64_t)A.result + (uint64_t)operation_map[j].result;
                        if (sum <= MAX_RESULT)
                        {
                            uint64_t key = pack_key((type)sum, combined);
                            bool need_insert = false;
                            {
                                std::lock_guard<std::mutex> lg(seen_mutex);
                                if (global_seen.find(key) == global_seen.end())
                                {
                                    global_seen.insert(key);
                                    need_insert = true;
                                }
                            }
                            if (need_insert)
                            {
                                operation op;
                                op.flags = combined;
                                op.result = (type)sum;
                                op.out = "(" + A.out + "+" + operation_map[j].out + ")";
                                worker_local_new.push_back(std::move(op));
                                ++produced;
                            }
                        }

                        // Multiplication
                        uint64_t prod = (uint64_t)A.result * (uint64_t)operation_map[j].result;
                        if (prod <= MAX_RESULT)
                        {
                            uint64_t key = pack_key((type)prod, combined);
                            bool need_insert = false;
                            {
                                std::lock_guard<std::mutex> lg(seen_mutex);
                                if (global_seen.find(key) == global_seen.end())
                                {
                                    global_seen.insert(key);
                                    need_insert = true;
                                }
                            }
                            if (need_insert)
                            {
                                operation op;
                                op.flags = combined;
                                op.result = (type)prod;
                                op.out = "(" + A.out + "*" + operation_map[j].out + ")";
                                worker_local_new.push_back(std::move(op));
                                ++produced;
                            }
                        }

                        // Exponentiation A^B
                        if (A.result >= 2 && operation_map[j].result >= 1)
                        {
                            uint64_t p = safe_pow_uint64(A.result, operation_map[j].result, MAX_RESULT);
                            if (p <= MAX_RESULT)
                            {
                                uint64_t key = pack_key((type)p, combined);
                                bool need_insert = false;
                                {
                                    std::lock_guard<std::mutex> lg(seen_mutex);
                                    if (global_seen.find(key) == global_seen.end())
                                    {
                                        global_seen.insert(key);
                                        need_insert = true;
                                    }
                                }
                                if (need_insert)
                                {
                                    operation op;
                                    op.flags = combined;
                                    op.result = (type)p;
                                    op.out = "(" + A.out + "^" + operation_map[j].out + ")";
                                    worker_local_new.push_back(std::move(op));
                                    ++produced;
                                }
                            }
                        }

                        // Exponentiation B^A
                        if (operation_map[j].result >= 2 && A.result >= 1)
                        {
                            uint64_t p = safe_pow_uint64(operation_map[j].result, A.result, MAX_RESULT);
                            if (p <= MAX_RESULT)
                            {
                                uint64_t key = pack_key((type)p, combined);
                                bool need_insert = false;
                                {
                                    std::lock_guard<std::mutex> lg(seen_mutex);
                                    if (global_seen.find(key) == global_seen.end())
                                    {
                                        global_seen.insert(key);
                                        need_insert = true;
                                    }
                                }
                                if (need_insert)
                                {
                                    operation op;
                                    op.flags = combined;
                                    op.result = (type)p;
                                    op.out = "(" + operation_map[j].out + "^" + A.out + ")";
                                    worker_local_new.push_back(std::move(op));
                                    ++produced;
                                }
                            }
                        }
                    }
                }
            }

            // Flush worker_local_new if big
            if (worker_local_new.size() >= FLUSH_BATCH)
            {
                {
                    std::lock_guard<std::mutex> lg(op_map_mutex);
                    for (size_t k = 0; k < worker_local_new.size() && operation_map.size() < max_ops; ++k)
                    {
                        operation_map.push_back(std::move(worker_local_new[k]));
                    }
                }
                worker_local_new.clear();
                if (operation_map.size() - last_saved_index.load(std::memory_order_acquire) >= SNAPSHOT_THRESHOLD)
                {
                    cv_snapshot.notify_one();
                }
            }

            // Early stop if reached max
            {
                std::lock_guard<std::mutex> lg(op_map_mutex);
                if (operation_map.size() >= max_ops)
                    break;
            }
        }

        // Final flush of worker_local_new
        if (!worker_local_new.empty())
        {
            {
                std::lock_guard<std::mutex> lg(op_map_mutex);
                for (size_t k = 0; k < worker_local_new.size() && operation_map.size() < max_ops; ++k)
                {
                    operation_map.push_back(std::move(worker_local_new[k]));
                }
            }
            worker_local_new.clear();
            if (operation_map.size() - last_saved_index.load(std::memory_order_acquire) >= SNAPSHOT_THRESHOLD)
            {
                cv_snapshot.notify_one();
            }
        }

        if (VERBOSE)
        {
            std::lock_guard<std::mutex> lg(log_mutex);
            std::clog << "[worker " << tid << "] produced=" << produced << " final_map_size=" << operation_map.size() << "\n";
        }
    };

    // Launch threads dividing i in roughly equal chunks over initial_frontier
    std::vector<std::thread> threads;
    size_t chunk = (initial_frontier + num_threads - 1) / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
        size_t i_start = t * chunk;
        size_t i_end = std::min(initial_frontier, i_start + chunk);
        if (i_start >= i_end)
            break;
        threads.emplace_back(worker, t, i_start, i_end);
    }

    // Join workers
    for (size_t t = 0; t < threads.size(); ++t)
    {
        threads[t].join();
    }

    // Request snapshot thread to finish and join
    snapshot_stop_requested = true;
    cv_snapshot.notify_all();
    if (snapshot_thread.joinable())
        snapshot_thread.join();

    if (VERBOSE)
    {
        std::lock_guard<std::mutex> lg(log_mutex);
        std::clog << "[build] Combine phase finished total_ops=" << operation_map.size() << "\n";
    }
}

#endif // DIGITS_CORE_HPP
