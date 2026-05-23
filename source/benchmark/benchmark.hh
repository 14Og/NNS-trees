#ifndef SOURCE_BENCHMARK_BENCHMARK_HH_
#define SOURCE_BENCHMARK_BENCHMARK_HH_

#include "generic/index.hh"
#include "generic/types.hh"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

class Benchmark {
public:
	Benchmark(std::unique_ptr<NNSIndex> aIndex, size_t aK, size_t aIters,
		float aNoise = 0.01f, uint64_t aSeed = 42) :
		index(std::move(aIndex)),
		k(aK),
		iters(aIters),
		noise(aNoise),
		seed(aSeed)
	{
	}

	void run(PointsPtr aPoints)
	{
		points = aPoints;

		// Build
		auto t0 = Clock::now();
		index->build(aPoints);
		buildMs = toMs(Clock::now() - t0);

		// Generate query points (random dataset point + Gaussian noise)
		auto queries = generateQueries();

		// Query
		std::vector<double> times, nodesVisited;
		times.reserve(iters);
		nodesVisited.reserve(iters);

		for (size_t i = 0; i < iters; ++i) {
			auto q0		= Clock::now();
			auto result = index->query(queries[i], k);
			auto q1		= Clock::now();

			times.push_back(toUs(q1 - q0));
			nodesVisited.push_back(static_cast<double>(result.nodesVisited));

			if (i == 0) {
				firstQuery	= queries[0];
				firstResult = std::move(result);
			}
		}

		queryMeanUs = mean(times);
		queryStdUs	= std(times, queryMeanUs);
		nodesMean	= mean(nodesVisited);
		nodesStd	= std(nodesVisited, nodesMean);
	}

	// Append one row to a stats CSV (writes header if file is new).
	void writeStats(
		const std::filesystem::path &aPath, const std::string &aAlgo) const
	{
		std::ofstream f(aPath);

		f << "algo,n,d,k,iters,"
			 "build_ms,query_mean_us,query_std_us,"
			 "nodes_mean,nodes_std\n";

		f << aAlgo << ',' << points->size() << ',' << points->at(0).size()
		  << ',' << k << ',' << iters << ',' << buildMs << ',' << queryMeanUs
		  << ',' << queryStdUs << ',' << nodesMean << ',' << nodesStd << '\n';
	}

	// Write neighbours of the first query point (for visualisation).
	// Format: idx,dist,x0,x1,...  — first row is the query point (idx=-1,
	// dist=0).
	void writeNeighbours(const std::filesystem::path &aPath) const
	{
		size_t d = points->at(0).size();
		std::ofstream f(aPath);

		f << "idx,dist";
		for (size_t i = 0; i < d; ++i) f << ",x" << i;
		f << '\n';

		f << "-1,0";
		for (float v : firstQuery) f << ',' << v;
		f << '\n';

		for (const auto &n : firstResult.neighbours) {
			f << n.idx << ',' << n.dist;
			for (auto v : points->at(n.idx)) f << ',' << v;
			f << '\n';
		}
	}

private:
	using Clock = std::chrono::high_resolution_clock;

	template<typename D>
	static double toMs(D aD)
	{
		return std::chrono::duration<double, std::milli>(aD).count();
	}

	template<typename D>
	static double toUs(D aD)
	{
		return std::chrono::duration<double, std::micro>(aD).count();
	}

	static double mean(const std::vector<double> &aV)
	{
		return std::accumulate(aV.begin(), aV.end(), 0.0) / aV.size();
	}

	static double std(const std::vector<double> &aV, double aMean)
	{
		double acc = 0.0;
		for (double x : aV) acc += (x - aMean) * (x - aMean);
		return std::sqrt(acc / aV.size());
	}

	Points generateQueries() const
	{
		std::mt19937 rng(seed);
		std::uniform_int_distribution<size_t> idxDist(0, points->size() - 1);
		std::normal_distribution<float> noiseDist(0.0f, noise);

		Points queries;
		queries.reserve(iters);
		for (size_t i = 0; i < iters; ++i) {
			Point q = points->at(idxDist(rng));
			for (auto &v : q) v += noiseDist(rng);
			queries.push_back(std::move(q));
		}
		return queries;
	}

	std::unique_ptr<NNSIndex> index;
	size_t k;
	size_t iters;
	float noise;
	uint64_t seed;
	PointsPtr points;

	double buildMs{0};
	double queryMeanUs{0};
	double queryStdUs{0};
	double nodesMean{0};
	double nodesStd{0};

	Point firstQuery;
	QueryResult firstResult;
};

#endif /* SOURCE_BENCHMARK_BENCHMARK_HH_ */