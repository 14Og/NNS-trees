#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "arg_helper.hh"
#include "benchmark.hh"
#include "dataloader.hh"
#include "generic/logger.hh"
#include "exhaustive_knn/exhaustive_knn.hh"
#include "kd_tree/kd_tree.hh"
#include "quad_tree/quad_tree.hh"

static inline std::filesystem::path dataDir{DATA_DIR};
static inline std::filesystem::path outDir{STATS_DIR};

static void printUsage()
{
	std::cerr
		<< "Usage: benchmark --data <filename> [options]\n"
		   "\n"
		   "  --data <filename>    CSV file name (relative to DATA_DIR)\n"
		   "  --algo <algo>        exhaustive (default) | kdtree | quadtree\n"
		   "  --k <int>            neighbours (default: 10)\n"
		   "  --iters <int>        query iterations (default: 500)\n"
		   "  --noise <float>      gaussian noise stddev (default: 0.01)\n"
		   "  --seed <int>         rng seed (default: 42)\n"
		   "  --save-neighbours    also write neighbours CSV for "
		   "visualisation\n"
		   "\n"
		   "Output (in STATS_DIR): <algo>_<stem>_stats.csv, "
		   "<algo>_<stem>_neighbours.csv\n";
}

int main(int argc, char **argv)
{
	std::span<char *> args(argv, argc);

	if (hasFlag(args, "--help") || hasFlag(args, "-h")) {
		printUsage();
		return 0;
	}

	auto dataFile = getArg(args, "--data");
	if (dataFile.empty()) {
		printUsage();
		return 1;
	}

	auto algo	   = getArg(args, "--algo", "exhaustive");
	size_t k	   = std::stoul(std::string(getArg(args, "--k", "10")));
	size_t iters   = std::stoul(std::string(getArg(args, "--iters", "500")));
	float noise	   = std::stof(std::string(getArg(args, "--noise", "0.01")));
	uint64_t seed  = std::stoull(std::string(getArg(args, "--seed", "42")));
	bool saveNeigh = hasFlag(args, "--save-neighbours");

	// Derive output paths: STATS_DIR/<algo>_<stem>_stats.csv
	std::filesystem::path dataPath = dataDir / dataFile;
	std::string stem			   = dataPath.stem().string();
	std::string prefix			   = std::string(algo) + "_" + stem;

	std::filesystem::create_directories(outDir);
	auto statsPath = outDir / (prefix + "_stats.csv");
	auto neighPath = outDir / (prefix + "_neighbours.csv");

	if constexpr (Logger::kEnabled)
		Logger::init(outDir / "logs" / (prefix + ".log"));

	// Instantiate index
	std::unique_ptr<NNSIndex> index;
	if (algo == "exhaustive")
		index = std::make_unique<ExhaustiveKNN>();
	else if (algo == "kdtree")
		index = std::make_unique<KDTree>();
	else if (algo == "quadtree")
		index = std::make_unique<QuadTree>();
	else {
		std::cerr << "Unknown algo: " << algo << '\n';
		return 1;
	}

	DataLoader loader{dataPath};
	Benchmark bench(std::move(index), k, iters, noise, seed);
	bench.run(loader.get());

	bench.writeStats(statsPath, std::string(algo));
	if (saveNeigh)
		bench.writeNeighbours(neighPath);

	std::cout << "stats -> " << statsPath << '\n';
	if (saveNeigh)
		std::cout << "neighbours -> " << neighPath << '\n';

	return 0;
}
