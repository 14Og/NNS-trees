#ifndef SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_
#define SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_

#include "../generic/index.hh"

// Flat linear-scan KNN.
//
// Build: O(1)  — just stores the data pointer.
// Query: O(n log k) — scans all n points, maintains a max-heap of size k.

// The max-heap keeps the k best candidates seen so far with the worst
// (largest distance) at the top, so eviction of a newly dominated candidate
// is O(log k).  Final results are returned sorted ascending by L2 distance.
class ExhaustiveKNN : public NNSIndex {
public:
	void build(PointsPtr aPoints) override;
	Neighbours query(const Point &aQ, size_t aK) const override;
};

#endif /* SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_ */
