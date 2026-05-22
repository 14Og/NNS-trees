#ifndef SOURCE_KD_TREE_KD_NODE_HH_
#define SOURCE_KD_TREE_KD_NODE_HH_

#include <memory>

struct KDNode;

using KDNodePtr = std::unique_ptr<KDNode>;

struct KDNode {
	size_t pointId{0}; // point index in input array
	KDNodePtr left;
	KDNodePtr right;
};

#endif /* SOURCE_KD_TREE_KD_NODE_HH_ */
