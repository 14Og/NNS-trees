#ifndef SOURCE_KD_TREE_KD_TREE_HH_
#define SOURCE_KD_TREE_KD_TREE_HH_

#include "generic/functions.hh"
#include "generic/index.hh"
#include "generic/logger.hh"
#include "kd_node.hh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>

class KDTree : public NNSIndex {
public:
	KDTree()						  = default;
	KDTree(const KDTree &)			  = delete;
	KDTree &operator=(const KDTree &) = delete;
	KDTree(KDTree &&)				  = delete;
	KDTree &operator=(KDTree &&)	  = delete;

	void build(PointsPtr aPoints) final
	{
		if (!aPoints || aPoints->empty())
			throw std::invalid_argument("KDTree::build: empty dataset");
		points = std::move(aPoints);

		std::vector<size_t> indices(points->size());
		std::iota(indices.begin(), indices.end(), 0);

		size_t d = points->at(0).size();
		root	 = Detail::build(*points, indices, 0, indices.size(), 0,
				Point(d, 0.0f), Point(d, 1.0f));
	}

	QueryResult query(const Point &aQ, size_t aK) const final
	{
		if (!points)
			throw std::runtime_error(
				"KDTree::query: call build() before query()");
		if (aK == 0)
			throw std::invalid_argument("KDTree::query: k must be positive");

		Logger::queryPoint(aQ);
		std::priority_queue<Neighbour> heap;
		size_t nodesVisited = 0;

		Detail::query(*points, root.get(), aQ, aK, 0, heap, nodesVisited);

		// Extract, sqrt squared distances, sort ascending.
		Neighbours result;
		result.reserve(heap.size());
		while (!heap.empty()) {
			auto n = heap.top();
			heap.pop();
			n.dist = std::sqrt(n.dist);
			result.push_back(n);
		}
		std::sort(result.begin(), result.end());
		for (const auto &n : result) Logger::result(n.idx, n.dist);
		return {std::move(result), nodesVisited};
	}

private:
	struct Detail {
		// Recursively builds the KD-tree over indices[lo, hi).
		// nth_element partitions in O(n) per level → O(n log n) total.
		static KDNodePtr build(const Points &aPoints,
			std::vector<size_t> &aIndices, size_t aLo, size_t aHi, int aDepth,
			Point aLoBound, Point aHiBound)
		{
			if (aLo >= aHi)
				return nullptr;

			int axis   = aDepth % static_cast<int>(aPoints[0].size());
			size_t mid = aLo + (aHi - aLo) / 2;

			std::nth_element(aIndices.begin() + aLo, aIndices.begin() + mid,
				aIndices.begin() + aHi, [&](size_t a, size_t b) {
					return aPoints[a][axis] < aPoints[b][axis];
				});

			auto node	   = std::make_unique<KDNode>();
			node->pointId  = aIndices[mid];

			float splitVal = aPoints[node->pointId][axis];
			Logger::split(node->pointId, axis, aDepth, aLoBound, aHiBound);

			Point leftHi  = aHiBound;
			leftHi[axis]  = splitVal;
			Point rightLo = aLoBound;
			rightLo[axis] = splitVal;

			node->left	  = build(
				   aPoints, aIndices, aLo, mid, aDepth + 1, aLoBound, leftHi);
			node->right = build(
				aPoints, aIndices, mid + 1, aHi, aDepth + 1, rightLo, aHiBound);
			return node;
		}

		// Traverses the tree maintaining a max-heap of the k best candidates
		// (squared L2).  Prunes subtrees whose closest possible point exceeds
		// the current worst heap entry.
		static void query(const Points &aPoints, const KDNode *aNode,
			const Point &aQ, size_t aK, int aDepth,
			std::priority_queue<Neighbour> &aHeap, size_t &aNodesVisited)
		{
			if (!aNode)
				return;

			++aNodesVisited;
			Logger::visit(aNode->pointId);

			// Update heap with this node's point (squared distance).
			auto dist = L2norm(aQ, aPoints[aNode->pointId]);
			if (aHeap.size() < aK) {
				aHeap.push({dist, aNode->pointId});
				Logger::accept(aNode->pointId, aHeap.top().dist);
			} else if (dist < aHeap.top().dist) {
				Logger::evict(aHeap.top().idx, aHeap.top().dist);
				aHeap.pop();
				aHeap.push({dist, aNode->pointId});
				Logger::accept(aNode->pointId, aHeap.top().dist);
			}

			// Signed distance to the splitting hyperplane:
			// negative → query is on the left, positive → right.
			int axis   = aDepth % static_cast<int>(aPoints[0].size());
			float diff = aQ[axis] - aPoints[aNode->pointId][axis];

			const KDNode *near
				= diff <= 0 ? aNode->left.get() : aNode->right.get();
			const KDNode *far
				= diff <= 0 ? aNode->right.get() : aNode->left.get();

			// Always visit the near side first.
			query(aPoints, near, aQ, aK, aDepth + 1, aHeap, aNodesVisited);

			// Visit far side only if it can contain closer points.
			// diff*diff is the squared distance to the splitting hyperplane.
			if (aHeap.size() < aK || diff * diff < aHeap.top().dist)
				query(aPoints, far, aQ, aK, aDepth + 1, aHeap, aNodesVisited);
			else if (far)
				Logger::prune(far->pointId, diff * diff);
		}
	};

private:
	KDNodePtr root;
};

#endif /* SOURCE_KD_TREE_KD_TREE_HH_ */
