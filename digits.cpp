/**
 * @file digits.cpp
 * @author LAPCoder
 *
 * The goal is to make a program that for any given number find an expression
 * such as:
 * - each digit (0-9) is used 1 time max
 * - Operations allowed: + - * ^ ()
 * - no intermediate values greater than the result
 *
 * New features:
 * - verbose mode (--verbose)
 * - build precalculated table (--build)
 * - save/load table to/from binary file (--save file.bin / --load file.bin)
 * - multithreading + snapshots
 * - final cleanup of useless parentheses
 * 
 * TODO:
 * - fix 0s at the start at numbers (like 11257 = 09573+1684)
 *   it must update flags, so find the origin of the bug
 * - SIMD
 * - GPU
 * - Multithreading when number is not in database
 */

#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstring>
#include <system_error>

// Store what digits are used (0-9) in one var.
// Order: oooo oo98 7654 3210
typedef uint16_t numUsed;

// The main type
typedef uint32_t type;

///------- Operations to compare to numUsed ---------------------------------///

// Return true if one or more digit is used 2 times (i.e. intersection != 0)
#define NUM_USED_2_TIMES(a, b) ((((a) & (b)) != 0))

// Return the combined used digits by 2 numUsed (OR)
#define COMBINED_NUMS(a, b) (((a) | (b)))

// Return the Nth bit
#define FLAG_OF_DIGIT(a, n) (((a) >> (n)) & 1)

///------- STRUCTS ----------------------------------------------------------///

struct operation
{
	numUsed flags;	 // What digits are used?
	type result;	 // Result of the OP
	std::string out; // The result in human-readable form.
};

///------- CONSTANTS --------------------------------------------------------///

constexpr type MAX_RESULT = 10'000'000; // Maximum number size
constexpr size_t MAX_PRECALC_OPS = 1'000'000'000; // Suggested reserve
const operation voidOP = {0, 0, ""};			// To initialize at the start

///------- GLOBALS ----------------------------------------------------------///

std::vector<operation> operation_map; // main storage (global)
bool VERBOSE = false;

///------- CLEANUP FUNCTIONS ------------------------------------------------///

// remove outer parentheses if the whole string is wrapped
static std::string strip_outer(const std::string &expr)
{
	if (expr.size() >= 2 && expr.front() == '(' && expr.back() == ')')
	{
		int depth = 0;
		for (size_t i = 0; i < expr.size(); i++)
		{
			if (expr[i] == '(')
				depth++;
			else if (expr[i] == ')')
				depth--;
			if (depth == 0 && i != expr.size() - 1)
				return expr; // not a single wrapping pair
		}
		return expr.substr(1, expr.size() - 2);
	}
	return expr;
}

// flatten associative ops: ((A+B)+C) => (A+B+C)
static std::string flatten(const std::string &expr, char op)
{
	std::string res;
	int depth = 0;
	for (size_t i = 0; i < expr.size(); i++)
	{
		char c = expr[i];
		if (c == '(')
			depth++;
		else if (c == ')')
			depth--;
		if (depth == 1 && c == op)
		{
			// remove parenthesis boundary around nested
			if (res.size() >= 2 && res.back() == ')')
				res.pop_back();
			res.push_back(op);
			continue;
		}
		res.push_back(c);
	}
	return res;
}

static void clean_expression(std::string &expr)
{
	bool changed = true;
	while (changed)
	{
		changed = false;
		std::string s = strip_outer(expr);
		if (s != expr)
		{
			expr = s;
			changed = true;
		}
		if (expr.find("+") != std::string::npos)
		{
			std::string f = flatten(expr, '+');
			if (f != expr)
			{
				expr = f;
				changed = true;
			}
		}
		if (expr.find("*") != std::string::npos)
		{
			std::string f = flatten(expr, '*');
			if (f != expr)
			{
				expr = f;
				changed = true;
			}
		}
	}
}

///------- THREAD-SAFE HELPERS -----------------------------------------------///

static inline uint64_t pack_key(type result, numUsed flags)
{
	return ((uint64_t)result << 16) | (uint64_t)flags;
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

static std::mutex op_map_mutex; // protect operation_map & size
static std::mutex seen_mutex;	// protect global_seen
static std::mutex log_mutex;	// protect logging (std::clog)
static std::condition_variable cv_snapshot;
static std::atomic<size_t> last_saved_index(0); // index up to which we already saved to disk
static bool snapshot_stop_requested = false;

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

// build_operation_map with multithreading + periodic snapshot writes
void build_operation_map(size_t max_ops = 5'000'000, int max_concat_len = 7)
{
	// Reserve to avoid reallocation (critical for safe concurrent reads/writes)
	{
		std::lock_guard<std::mutex> lg(op_map_mutex);
		if (operation_map.capacity() < max_ops)
			operation_map.reserve(max_ops);
	}

	// global seen set for dedupe
	static std::unordered_set<uint64_t> global_seen;
	{
		std::lock_guard<std::mutex> lg(seen_mutex);
		global_seen.clear();
		global_seen.reserve(std::min<uint64_t>((uint64_t)max_ops * 2ULL, 100000000ULL));
		// initialize seen with existing operation_map
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

	// local function: generate concatenations (single-threaded)
	// This populates operation_map (thread-safe) and global_seen under locks
	std::function<void(uint16_t, int, type, std::string &)> dfs_concat;
	dfs_concat = [&](uint16_t flags, int depth, type current_val, std::string &repr)
	{
		{
			std::lock_guard<std::mutex> lg(op_map_mutex);
			if (operation_map.size() >= max_ops)
				return;
		}
		if (depth > max_concat_len)
			return;

		if (!repr.empty())
		{
			if ((current_val > 0 && current_val <= MAX_RESULT) || (current_val == 0 && repr.size() == 1))
			{
				uint64_t key = pack_key(current_val, flags);
				bool inserted = false;
				{
					std::lock_guard<std::mutex> lg(seen_mutex);
					if (global_seen.insert(key).second)
						inserted = true;
				}
				if (inserted)
				{
					operation o;
					o.flags = flags;
					o.result = current_val;
					o.out = repr;
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
				{
					std::lock_guard<std::mutex> lg(op_map_mutex);
					if (operation_map.size() >= max_ops)
						return;
				}
			}
		}

		if (depth == max_concat_len)
			return;

		for (int d = 0; d <= 9; ++d)
		{
			{
				std::lock_guard<std::mutex> lg(op_map_mutex);
				if (operation_map.size() >= max_ops)
					return;
			}
			if (FLAG_OF_DIGIT(flags, d))
				continue;
			// avoid leading zero multi-digit
			if (repr.empty() && d == 0)
			{
				std::string tmp("0");
				uint16_t nf = flags | (1u << d);
				std::string saved = repr;
				repr = tmp;
				dfs_concat(nf, depth + 1, 0, repr);
				repr = saved;
				continue;
			}
			uint64_t nv = (uint64_t)current_val * 10ULL + (uint64_t)d;
			if (nv > MAX_RESULT)
				continue;
			std::string saved = repr;
			if (repr.empty())
				repr = std::to_string(d);
			else
				repr.push_back(char('0' + d));
			uint16_t nf = flags | (1u << d);
			dfs_concat(nf, depth + 1, (type)nv, repr);
			repr = saved;
			{
				std::lock_guard<std::mutex> lg(op_map_mutex);
				if (operation_map.size() >= max_ops)
					return;
			}
		}
	};

	// run concatenation single-threaded (fast)
	{
		std::string rep;
		dfs_concat(0, 0, 0, rep);
	}

	if (VERBOSE)
	{
		std::lock_guard<std::mutex> lg(log_mutex);
		std::clog << "[build] Concatenation phase finished. ops=" << operation_map.size() << "\n";
	}

	// Now combine pairs in parallel.
	// We'll spawn worker threads. Each thread processes a range of i in [0, frontier_end).
	// Each thread collects local_new_ops and then flushes them under lock in batches to reduce contention.

	const size_t initial_frontier = operation_map.size();
	if (initial_frontier == 0)
	{
		if (VERBOSE)
		{
			std::lock_guard<std::mutex> lg(log_mutex);
			std::clog << "[build] nothing to combine, leaving\n";
		}
		return;
	}

	// thread params (tunable)
	unsigned int hw = std::thread::hardware_concurrency();
	unsigned int num_threads = hw > 0 ? hw : 4;
	const size_t MAX_WINDOW = 200000;
	const size_t FLUSH_BATCH = 4096;		  // flush local results every FLUSH_BATCH collected
	const size_t SNAPSHOT_THRESHOLD = 250000; // trigger snapshot when new entries >= threshold
	const size_t SNAPSHOT_INTERVAL_SEC = 10;  // snapshot thread wake interval

	if (VERBOSE)
	{
		std::lock_guard<std::mutex> lg(log_mutex);
		std::clog << "[build] Starting pairwise combine with " << num_threads << " threads\n";
	}

	// snapshot file base (part + rename). We'll notify snapshot thread via cv_snapshot when threshold passed.
	const std::string snapshot_file = "precalc.bin"; // default name; main will rename later as required
	const std::string snapshot_part = snapshot_file + ".part";

	// Start snapshot thread
	snapshot_stop_requested = false;
	last_saved_index.store(0, std::memory_order_release);
	std::thread snapshot_thread(snapshot_thread_func, snapshot_file, snapshot_part,
								SNAPSHOT_INTERVAL_SEC, SNAPSHOT_THRESHOLD);

	// worker thread lambda
	auto worker = [&](unsigned int tid, size_t i_start, size_t i_end)
	{
		std::vector<operation> local_new;
		local_new.reserve(FLUSH_BATCH * 2);

		size_t produced = 0;
		if (VERBOSE)
		{
			std::lock_guard<std::mutex> lg(log_mutex);
			std::clog << "[worker " << tid << "] starting i=" << i_start << " i_end=" << i_end << "\n";
		}

		for (size_t i = i_start; i < i_end; ++i)
		{
			// check global stop conditions
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
			for (size_t j = i; j < jmax; ++j)
			{
				if (NUM_USED_2_TIMES(A.flags, operation_map[j].flags))
					continue;
				numUsed combined = COMBINED_NUMS(A.flags, operation_map[j].flags);

				// addition
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
						local_new.push_back(std::move(op));
						++produced;
					}
				}

				// multiplication
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
						local_new.push_back(std::move(op));
						++produced;
					}
				}

				// exponent A^B
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
							local_new.push_back(std::move(op));
							++produced;
						}
					}
				}
				// exponent B^A
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
							local_new.push_back(std::move(op));
							++produced;
						}
					}
				}

				// flush local_new if big
				if (local_new.size() >= FLUSH_BATCH)
				{
					// flush to global op_map
					{
						std::lock_guard<std::mutex> lg(op_map_mutex);
						for (size_t k = 0; k < local_new.size() && operation_map.size() < max_ops; ++k)
						{
							operation_map.push_back(std::move(local_new[k]));
						}
					}
					local_new.clear();
					// notify snapshot thread if threshold passed
					if (operation_map.size() - last_saved_index.load(std::memory_order_acquire) >= SNAPSHOT_THRESHOLD)
					{
						cv_snapshot.notify_one();
					}
				}

				// early stop if reached max
				{
					std::lock_guard<std::mutex> lg(op_map_mutex);
					if (operation_map.size() >= max_ops)
						break;
				}
			} // j
		} // i

		// final flush of local_new
		if (!local_new.empty())
		{
			{
				std::lock_guard<std::mutex> lg(op_map_mutex);
				for (size_t k = 0; k < local_new.size() && operation_map.size() < max_ops; ++k)
				{
					operation_map.push_back(std::move(local_new[k]));
				}
			}
			local_new.clear();
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
	}; // worker

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

	// join workers
	for (size_t t = 0; t < threads.size(); ++t)
	{
		threads[t].join();
	}

	// request snapshot thread to finish and join
	snapshot_stop_requested = true;
	cv_snapshot.notify_all();
	if (snapshot_thread.joinable())
		snapshot_thread.join();

	if (VERBOSE)
	{
		std::lock_guard<std::mutex> lg(log_mutex);
		std::clog << "[build] combine phase finished total_ops=" << operation_map.size() << "\n";
	}
}

///------- CLI / main ----------------------------------------------------------///

static void print_usage(const char *prog)
{
	std::cerr << "Usage: " << prog << " [--build] [--load file.bin] [--save file.bin] [--verbose] [number]\n";
	std::cerr << "  --build            Build precalculated table\n";
	std::cerr << "  --load <file>      Load table from binary file\n";
	std::cerr << "  --save <file>      Save table to binary file (final full save)\n";
	std::cerr << "  --verbose          Verbose output\n";
	std::cerr << "  [number]           Optional: find expression for number\n";
}

int main(int argc, char **argv)
{
	bool do_build = false;
	std::string load_file;
	std::string save_file;
	VERBOSE = false;
	std::vector<std::string> extras;
	int max_concat_len = 14; // default

	for (int i = 1; i < argc; ++i)
	{
		std::string a = argv[i];
		if (a == "--build")
			do_build = true;
		else if (a == "--verbose")
			VERBOSE = true;
		else if (a == "--load")
		{
			if (i + 1 < argc)
				load_file = argv[++i];
			else
			{
				std::cerr << "--load requires filename\n";
				return 1;
			}
		}
		else if (a == "--save")
		{
			if (i + 1 < argc)
				save_file = argv[++i];
			else
			{
				std::cerr << "--save requires filename\n";
				return 1;
			}
		}
		else if (a == "--help" || a == "-h")
		{
			print_usage(argv[0]);
			return 0;
		}
		else if (a == "--depth")
		{
			if (i + 1 < argc)
				max_concat_len = std::stoi(argv[++i]);
			else
			{
				std::cerr << "--depth requires an integer\n";
				return 1;
			}
		}
		else
		{
			extras.push_back(a);
		}
	}

	// Optionally load first
	if (!load_file.empty())
	{
		if (VERBOSE)
			std::clog << "[main] Loading from '" << load_file << "' ...\n";
		if (!load_operation_map(load_file))
		{
			std::cerr << "Failed to load from file: " << load_file << "\n";
			return 2;
		}
		if (VERBOSE)
			std::clog << "[main] Loaded " << operation_map.size() << " operations.\n";
	}

	// If build requested, (re)build
	if (do_build)
	{
		size_t desired_ops = 50'000'000;
		if (VERBOSE)
			std::clog << "[main] Building operation_map (target ops ~" << desired_ops
					  << ", depth=" << max_concat_len << ")...\n";
		build_operation_map(desired_ops, max_concat_len);
		if (VERBOSE)
			std::clog << "[main] Build completed: total ops=" << operation_map.size() << "\n";
	}

	// Optionally save final full map (user requested)
	if (!save_file.empty())
	{
		if (VERBOSE)
			std::clog << "[main] Saving final full map to '" << save_file << "' ...\n";
		if (!save_operation_map_full(save_file))
		{
			std::cerr << "Failed to save to file: " << save_file << "\n";
			return 3;
		}
		if (VERBOSE)
			std::clog << "[main] Saved " << operation_map.size() << " operations.\n";
	}

	// If a number is provided, try to search it
	if (!extras.empty())
	{
		try
		{
			long long t = std::stoll(extras[0]);
			if (t < 0 || t > (long long)MAX_RESULT)
			{
				std::cerr << "Target must be in [0, " << MAX_RESULT << "]\n";
				return 4;
			}
			type target = (type)t;
			if (VERBOSE)
				std::clog << "[main] Searching expression for " << target << " ...\n";
			operation found;
			if (search_op(found, target))
			{
				clean_expression(found.out);
				std::cout << target << " = " << found.out << "\n";
			}
			else
			{
				std::cout << "No expression found for " << target << "\n";
			}
		}
		catch (...)
		{
			std::cerr << "Invalid number: " << extras[0] << "\n";
			return 5;
		}
	}
	else
	{
		if (VERBOSE)
			std::clog << "[main] No number provided. Exiting.\n";
	}

	return 0;
}
