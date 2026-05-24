#ifndef SOURCE_GENERIC_LOGGER_HH_
#define SOURCE_GENERIC_LOGGER_HH_

#include "types.hh"

#include <filesystem>
#include <fstream>
#include <stdexcept>

// Compile-time logger — all methods compile to nothing unless ENABLE_LOGGING
// is defined.  Use the `logging` CMake preset to get a logging build;
// the `release` preset is always logging-free with zero overhead.
//
// Call Logger::init(path) once before build()/query() to open the log file.
struct Logger {
	static constexpr bool kEnabled =
#ifdef ENABLE_LOGGING
		true
#else
		false
#endif
		;

	static void init(const std::filesystem::path &aPath)
	{
		if constexpr (kEnabled) {
			std::filesystem::create_directories(aPath.parent_path());
			file().open(aPath);
			if (!file())
				throw std::runtime_error(
					"Logger::init: cannot open " + aPath.string());
		}
	}

	template<typename... Args>
	static void log(Args &&...aArgs)
	{
		if constexpr (kEnabled)
			(file() << ... << aArgs) << '\n';
		else
			((void)aArgs, ...);
	}

	// Build events.
	// Bounds are the cell's [lo, hi) box inherited from the parent split.
	// Format: SPLIT idx axis depth lo0 lo1 ... hi0 hi1 ...
	static void split(size_t aIdx, int aAxis, int aDepth, const Point &aLoBound,
		const Point &aHiBound)
	{
		if constexpr (kEnabled) {
			file() << "SPLIT " << aIdx << ' ' << aAxis << ' ' << aDepth;
			for (float v : aLoBound) file() << ' ' << v;
			for (float v : aHiBound) file() << ' ' << v;
			file() << '\n';
		}
	}

	// Query point — logged once at the start of each query.
	// Format: QUERY x0 x1 ...
	static void queryPoint(const Point &aQ)
	{
		if constexpr (kEnabled) {
			file() << "QUERY";
			for (float v : aQ) file() << ' ' << v;
			file() << '\n';
		}
	}

	// Query events
	static void visit(size_t aIdx)
	{
		log("VISIT ", aIdx);
	}
	static void accept(size_t aIdx, float aDist)
	{
		log("ACCEPT ", aIdx, ' ', aDist);
	}
	static void evict(size_t aIdx, float aDist)
	{
		log("EVICT ", aIdx, ' ', aDist);
	}
	static void prune(size_t aIdx, float aDiffSq)
	{
		log("PRUNE ", aIdx, ' ', aDiffSq);
	}
	static void result(size_t aIdx, float aDist)
	{
		log("RESULT ", aIdx, ' ', aDist);
	}

private:
	static std::ofstream &file()
	{
		static std::ofstream stream;
		return stream;
	}
};

#endif /* SOURCE_GENERIC_LOGGER_HH_ */
