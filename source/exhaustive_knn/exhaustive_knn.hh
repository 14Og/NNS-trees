#ifndef SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_
#define SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_

#include "generic/functions.hh"
#include "generic/index.hh"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

// Flat linear-scan KNN.
//
// Build: O(1)  — just stores the data pointer.
// Query: O(n log k) — scans all n points, maintains a max-heap of size k.

// The max-heap keeps the k best candidates seen so far with the worst
// (largest distance) at the top, so eviction of a newly dominated candidate
// is O(log k).  Final results are returned sorted ascending by L2 distance.
class ExhaustiveKNN : public NNSIndex {
public:
	void build(PointsPtr aPoints) override
	{
		if (!aPoints || aPoints->empty())
			throw std::invalid_argument("ExhaustiveKNN::build: empty dataset");
		points = std::move(aPoints);
	}

	QueryResult query(const Point &aQ, size_t aK) const override
	{
		if (!points)
			throw std::runtime_error(
				"ExhaustiveKNN::query: call build() before query()");

		if (aK == 0)
			throw std::invalid_argument(
				"ExhaustiveKNN::query: call for positive number of neighbours");

		// Distances stored in the heap are *squared* L2 to avoid sqrt on every
		// comparison.  We take sqrt only on the k final winners below.
		std::priority_queue<Neighbour> heap;

		for (size_t i = 0; i < points->size(); ++i) {
			auto norm = L2norm(aQ, (*points)[i]);

			if (heap.size() < aK) {
				heap.push({norm, i});
			} else if (norm < heap.top().dist) {
				heap.pop();
				heap.push({norm, i});
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
		return {std::move(neighbours), points->size()};
	}
};

#endif /* SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_ */
