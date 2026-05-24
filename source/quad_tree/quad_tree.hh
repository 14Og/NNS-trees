#ifndef SOURCE_QUAD_TREE_QUAD_TREE_HH_
#define SOURCE_QUAD_TREE_QUAD_TREE_HH_

#include "generic/functions.hh"
#include "generic/index.hh"
#include "generic/logger.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

class QuadTree : public NNSIndex {
public:
	QuadTree()							  = default;
	QuadTree(const QuadTree &)			  = delete;
	QuadTree &operator=(const QuadTree &) = delete;
	QuadTree(QuadTree &&)				  = delete;
	QuadTree &operator=(QuadTree &&)	  = delete;

	void build(PointsPtr aPoints) final
	{
		if (!aPoints || aPoints->empty())
			throw std::invalid_argument("QuadTree::build: empty dataset");
		if (aPoints->at(0).size() != 2)
			throw std::invalid_argument(
				"QuadTree::build: only 2D points are supported");

		points = std::move(aPoints);

		std::vector<size_t> indices(points->size());
		for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

		root = Detail::build(*points, indices, {0.0f, 0.0f}, {1.0f, 1.0f}, 0);
	}

	QueryResult query(const Point &aQ, size_t aK) const final
	{
		if (!points)
			throw std::runtime_error(
				"QuadTree::query: call build() before query()");
		if (aK == 0)
			throw std::invalid_argument("QuadTree::query: k must be positive");
		if (aQ.size() != 2)
			throw std::invalid_argument(
				"QuadTree::query: only 2D points are supported");

		Logger::queryPoint(aQ);
		std::priority_queue<Neighbour> heap;
		size_t pointsVisited = 0;

		Detail::query(*points, root.get(), aQ, aK, heap, pointsVisited);

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
		return {std::move(result), pointsVisited};
	}

private:
	struct Node {
		Point lo;
		Point hi;
		std::vector<size_t> pointIds;
		std::array<std::unique_ptr<Node>, 4> children;

		bool isLeaf() const
		{
			return std::all_of(children.begin(), children.end(),
				[](const auto &aChild) { return aChild == nullptr; });
		}
	};

	struct Detail {
		static constexpr size_t kLeafCapacity = 8;
		static constexpr size_t kMaxDepth	  = 32;

		static std::unique_ptr<Node> build(const Points &aPoints,
			const std::vector<size_t> &aIndices, Point aLo, Point aHi,
			size_t aDepth)
		{
			auto node = std::make_unique<Node>();
			node->lo  = std::move(aLo);
			node->hi  = std::move(aHi);

			if (aIndices.size() <= kLeafCapacity || aDepth >= kMaxDepth) {
				node->pointIds = aIndices;
				return node;
			}

			Logger::splitQuad(static_cast<int>(aDepth), node->lo, node->hi);

			std::array<std::vector<size_t>, 4> buckets;
			for (auto idx : aIndices)
				buckets[quadrant(aPoints[idx], node->lo, node->hi)].push_back(
					idx);

			for (size_t q = 0; q < buckets.size(); ++q) {
				if (buckets[q].empty())
					continue;
				auto [lo, hi]	  = childBounds(node->lo, node->hi, q);
				node->children[q] = build(aPoints, buckets[q], std::move(lo),
					std::move(hi), aDepth + 1);
			}

			return node;
		}

		static void query(const Points &aPoints, const Node *aNode,
			const Point &aQ, size_t aK, std::priority_queue<Neighbour> &aHeap,
			size_t &aPointsVisited)
		{
			if (!aNode)
				return;

			const float boxDist = boxDistSq(aQ, aNode->lo, aNode->hi);
			if (aHeap.size() == aK && boxDist >= aHeap.top().dist) {
				Logger::pruneQuad(boxDist, aNode->lo, aNode->hi);
				return;
			}

			if (aNode->isLeaf()) {
				for (auto idx : aNode->pointIds) {
					++aPointsVisited;
					Logger::visit(idx);

					const float dist = L2norm(aQ, aPoints[idx]);
					if (aHeap.size() < aK) {
						aHeap.push({dist, idx});
						Logger::accept(idx, aHeap.top().dist);
					} else if (dist < aHeap.top().dist) {
						Logger::evict(aHeap.top().idx, aHeap.top().dist);
						aHeap.pop();
						aHeap.push({dist, idx});
						Logger::accept(idx, aHeap.top().dist);
					}
				}
				return;
			}

			std::array<const Node *, 4> children{};
			size_t count = 0;
			for (const auto &child : aNode->children) {
				if (child)
					children[count++] = child.get();
			}

			std::sort(children.begin(), children.begin() + count,
				[&](const Node *a, const Node *b) {
					return boxDistSq(aQ, a->lo, a->hi)
						< boxDistSq(aQ, b->lo, b->hi);
				});

			for (size_t i = 0; i < count; ++i)
				query(aPoints, children[i], aQ, aK, aHeap, aPointsVisited);
		}

		static size_t quadrant(
			const Point &aPoint, const Point &aLo, const Point &aHi)
		{
			const float midX = (aLo[0] + aHi[0]) * 0.5f;
			const float midY = (aLo[1] + aHi[1]) * 0.5f;
			const bool right = aPoint[0] >= midX;
			const bool top	 = aPoint[1] >= midY;
			return static_cast<size_t>(right) + 2 * static_cast<size_t>(top);
		}

		static std::pair<Point, Point> childBounds(
			const Point &aLo, const Point &aHi, size_t aQuadrant)
		{
			Point lo		 = aLo;
			Point hi		 = aHi;
			const float midX = (aLo[0] + aHi[0]) * 0.5f;
			const float midY = (aLo[1] + aHi[1]) * 0.5f;

			if (aQuadrant & 1)
				lo[0] = midX;
			else
				hi[0] = midX;

			if (aQuadrant & 2)
				lo[1] = midY;
			else
				hi[1] = midY;

			return {lo, hi};
		}

		static float boxDistSq(
			const Point &aQ, const Point &aLo, const Point &aHi)
		{
			float dist = 0.0f;
			for (size_t i = 0; i < 2; ++i) {
				if (aQ[i] < aLo[i]) {
					const float d = aLo[i] - aQ[i];
					dist += d * d;
				} else if (aQ[i] > aHi[i]) {
					const float d = aQ[i] - aHi[i];
					dist += d * d;
				}
			}
			return dist;
		}
	};

	std::unique_ptr<Node> root;
};

#endif /* SOURCE_QUAD_TREE_QUAD_TREE_HH_ */
