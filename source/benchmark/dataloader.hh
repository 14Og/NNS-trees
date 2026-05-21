#ifndef SOURCE_BENCHMARK_DATALOADER_HH_
#define SOURCE_BENCHMARK_DATALOADER_HH_

#include "generic/types.hh"

#include <filesystem>
#include <fstream>
#include <string>

class DataLoader {

public:
	DataLoader(std::filesystem::path aPath) : pointsPath(std::move(aPath))
	{
		if (!std::filesystem::exists(pointsPath))
			throw std::invalid_argument(
				"DataLoader::DataLoader: invalid CSV path");
	}

	PointsPtr get()
	{
		std::ifstream file(pointsPath);
		Points points;
		std::string line;
		while (std::getline(file, line)) {
			Point point;
			std::istringstream ss(line);
			std::string val;
			while (std::getline(ss, val, ',')) point.push_back(std::stof(val));
			points.push_back(std::move(point));
		}
		return std::make_shared<Points>(std::move(points));
	}

private:
	std::filesystem::path pointsPath;
};

#endif /* SOURCE_BENCHMARK_DATALOADER_HH_ */
