#ifndef SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_
#define SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_

#include "generic/functions.hh"
#include "generic/index.hh"
#include "generic/logger.hh"

#include <algorithm>
#include <cmath>
#include <queue>

// Flat linear-scan KNN.
//
// Build: O(1)  — just stores the data pointer.
// Query: O(n log k) — scans all n points, maintains a max-heap of size k.

// The max-heap keeps the k best candidates seen so far with the worst
// (largest distance) at the top, so eviction of a newly dominated candidate
// is O(log k).  Final results are returned sorted ascending by L2 distance.
class ExhaustiveKNN : public NNSIndex {
public:
	ExhaustiveKNN()									= default;
	ExhaustiveKNN(const ExhaustiveKNN &)			= delete;
	ExhaustiveKNN &operator=(const ExhaustiveKNN &) = delete;
	ExhaustiveKNN(ExhaustiveKNN &&)					= delete;
	ExhaustiveKNN &operator=(ExhaustiveKNN &&)		= delete;

	void build(PointsPtr aPoints) final
	{
		if (!aPoints || aPoints->empty())
			throw std::invalid_argument("ExhaustiveKNN::build: empty dataset");
		points = std::move(aPoints);
	}

	QueryResult query(const Point &aQ, size_t aK) const final
	{
		if (!points)
			throw std::runtime_error(
				"ExhaustiveKNN::query: call build() before query()");

		if (aK == 0)
			throw std::invalid_argument(
				"ExhaustiveKNN::query: call for positive number of neighbours");

		// Distances stored in the heap are *squared* L2 to avoid sqrt on every
		// comparison.  We take sqrt only on the k final winners below.
		Logger::queryPoint(aQ);
		std::priority_queue<Neighbour> heap;

		for (size_t i = 0; i < points->size(); ++i) {
			auto norm = L2norm(aQ, (*points)[i]);
			Logger::visit(i);

			if (heap.size() < aK) {
				heap.push({norm, i});
				Logger::accept(i, heap.top().dist);
			} else if (norm < heap.top().dist) {
				Logger::evict(heap.top().idx, heap.top().dist);
				heap.pop();
				heap.push({norm, i});
				Logger::accept(i, heap.top().dist);
			}
		}

		// Extract, convert squared distances to actual L2, sort ascending.
		Neighbours neighbours;
		neighbours.reserve(heap.size());
		while (!heap.empty()) {
			auto n = heap.top();
			heap.pop();
			n.dist = std::sqrt(n.dist);
			neighbours.push_back(n);
		}
		std::sort(neighbours.begin(), neighbours.end());
		for (const auto &n : neighbours) Logger::result(n.idx, n.dist);
		return {std::move(neighbours), points->size()};
	}
};

#endif /* SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_ */
