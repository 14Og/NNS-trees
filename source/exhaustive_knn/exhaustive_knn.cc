#include "exhaustive_knn.hh"
#include "../generic/functions.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <queue>
#include <stdexcept>

void ExhaustiveKNN::build(PointsPtr aPoints)
{
	if (!aPoints || aPoints->empty())
		throw std::invalid_argument("ExhaustiveKNN::build: empty dataset");
	points = std::move(aPoints);
}

Neighbours ExhaustiveKNN::query(const Point &aQ, size_t aK) const
{
	assert(points && "call build() before query()");
	assert(aK > 0);

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
	Neighbours result;
	result.reserve(heap.size());
	while (!heap.empty()) {
		auto n = heap.top();
		heap.pop();
		n.dist = std::sqrt(n.dist);
		result.push_back(n);
	}
	std::sort(result.begin(), result.end()); // uses Neighbour::operator<
	return result;
}
