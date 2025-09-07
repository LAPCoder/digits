#pragma once

#ifndef INCLUDES_HPP
#define INCLUDES_HPP

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

constexpr type MAX_RESULT = 276'447'232; // Maximum number size
constexpr size_t MAX_PRECALC_OPS = 10'000'000'000; // Suggested reserve
const operation voidOP = {0, 0, ""};			// To initialize at the start

///------- GLOBALS ----------------------------------------------------------///

extern std::vector<operation> operation_map; // main storage (global)
extern bool VERBOSE;

#endif // INCLUDES_HPP
