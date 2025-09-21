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
 * - fix the () not properly cleaned (#1)
 *   it must update flags, so find the origin of the bug
 * - GPU
 * - Multithreading when number is not in database
 * - allow +2 for input
 */

/* COMPILATION COMMANDS
BASIC:
g++ digits.cpp -o digits -Wall -Wextra -fuse-ld=lld -Wshadow -g3 -fsanitize=address,undefined -O3 -std=c++23 -march=native
SIMD (recommended):
g++ digits.cpp -o digits -Wall -Wextra -fuse-ld=lld -Wshadow -g3 -fsanitize=address,undefined -O3 -std=c++23 -march=native -DUSE_SIMD
NVIDIA GPU (NOT WORKING):
g++ digits.cpp -o digits.o -c -Wall -Wextra -fuse-ld=lld -Wshadow -g3 -O3 -std=c++23 -march=native -DUSE_GPU
nvcc -O3 -std=c++17 -c digits_gpu.cu -o digits_gpu.o -DUSE_GPU
nvcc -O3 -std=c++17 digits.o digits_gpu.o -o digits_gpu
--------------------------
First time:
./digits --build --save precalc.bin --verbose --depth 16 1
Then:
./digits --load precalc.bin --verbose 12234
--------------------------
To use with Emscripten:
em++ digits.cpp -o demo/digits.js -Wall -Wextra -Wshadow -O3 -std=c++23 -sEXPORTED_FUNCTIONS=_main,_input -sFORCE_FILESYSTEM -sINITIAL_MEMORY=256MB -sMAXIMUM_MEMORY=2GB -sALLOW_MEMORY_GROWTH -sEXPORTED_RUNTIME_METHODS=ccall,FS
*/

// g++ digits.cpp -o digits -Wall -Wextra -fuse-ld=lld -Wshadow -g3 -fsanitize=address,undefined -O3 -std=c++23 -march=native
// Add digits_gpu.o if you use the GPU
#include "includes.hpp"
#include "libs/exprtk.hpp"
#ifdef USE_GPU
#include "digits_gpu.hpp"
#include "digits_simd.hpp"
#elifdef USE_SIMD
#include "digits_simd.hpp"
#else
#include "digits_cpu.hpp"
#endif

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

static int eval(std::string in)
{
	try
	{
		typedef exprtk::symbol_table<double> symbol_table_t;
		typedef exprtk::expression<double> expression_t;
		typedef exprtk::parser<double> parser_t;

		symbol_table_t symbol_table;
		expression_t expression;
		expression.register_symbol_table(symbol_table);

		parser_t parser;
		parser.compile(in, expression);

		long long t = static_cast<long long>(expression.value());

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
		std::cerr << "Invalid number: " << in << "\n";
		return 5;
	}

	return 0;
}
#ifdef __EMSCRIPTEN__
#include <emscripten.h>

extern "C"
{
	EMSCRIPTEN_KEEPALIVE
	void input(const char *input)
	{
		std::string in(input);
		eval(in);
	}
}

extern "C"
{
	EMSCRIPTEN_KEEPALIVE
	int main()
	{
		std::cout << "[main] Emscripten mode (browser)\n";
		std::string load_file;
		VERBOSE = true;
		int max_concat_len = 14; // default

		if (!load_operation_map("precalc.bin"))
		{
			std::cerr << "[main] Can't load precalc.bin\n";
		}

		// Try to load the file that should be included
		/*
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
		}*/
		std::cout << "[main] Enter an expression: ";
		return 0;
	}
}
#else
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
		size_t desired_ops = 5'000'000;
		if (VERBOSE)
			std::clog << "[main] Building operation_map (target ops ~" << desired_ops
					  << ", depth=" << max_concat_len << ")...\n";

#ifdef USE_GPU
		build_operation_map_gpu(desired_ops, max_concat_len); // TODO better integration
#else
		build_operation_map(desired_ops, max_concat_len);
#endif
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
		eval(extras[0]);
	}

	// Enter in interactive mode: the user wil give a number, the programm will
	// give an answer

	while (true)
	{
		std::cout << "[main] Enter an expression (enter 'exit' to exit): ";
		std::string in;
		std::getline(std::cin, in);
		if (in == "exit")
			return 0;
		eval(in);
	}

	return -1;
}
#endif
