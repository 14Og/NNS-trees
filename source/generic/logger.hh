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
	// KD-Tree: pivot point splits the cell along one axis.
	// Format: SPLIT idx axis depth lo0 lo1 ... hi0 hi1 ...
	static void splitKD(
		size_t aIdx, int aAxis, int aDepth, const Point &aLo, const Point &aHi)
	{
		if constexpr (kEnabled) {
			file() << "SPLIT " << aIdx << ' ' << aAxis << ' ' << aDepth;
			for (float v : aLo) file() << ' ' << v;
			for (float v : aHi) file() << ' ' << v;
			file() << '\n';
		}
	}

	// QuadTree: cell is subdivided at its geometric midpoint into quadrants.
	// Format: QUAD depth lo0 lo1 hi0 hi1
	static void splitQuad(int aDepth, const Point &aLo, const Point &aHi)
	{
		if constexpr (kEnabled) {
			file() << "QUAD " << aDepth;
			for (float v : aLo) file() << ' ' << v;
			for (float v : aHi) file() << ' ' << v;
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
	// KD-Tree: far subtree pruned; idx is the subtree root's point.
	static void pruneKD(size_t aIdx, float aDiffSq)
	{
		log("PRUNE ", aIdx, ' ', aDiffSq);
	}

	// QuadTree: cell pruned because box distance exceeds heap worst.
	// Format: PRUNEBOX distSq lo0 lo1 hi0 hi1
	static void pruneQuad(float aDistSq, const Point &aLo, const Point &aHi)
	{
		if constexpr (kEnabled) {
			file() << "PRUNEBOX " << aDistSq;
			for (float v : aLo) file() << ' ' << v;
			for (float v : aHi) file() << ' ' << v;
			file() << '\n';
		}
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
