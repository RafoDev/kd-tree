// Copyright

#ifndef SRC_KDTREE_HPP_
#define SRC_KDTREE_HPP_

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#include "Point.hpp"

template <size_t N, typename ElemType>
class KDTree
{

public:
  typedef std::pair<Point<N>, ElemType> value_type;

  KDTree();

  ~KDTree();

  KDTree(const KDTree &rhs);

  KDTree &operator=(const KDTree &rhs);

  size_t dimension() const;

  size_t size() const;
  bool empty() const;

  bool contains(const Point<N> &pt) const;

  void insert(const Point<N> &pt, const ElemType &value);

  ElemType &operator[](const Point<N> &pt);

  ElemType &at(const Point<N> &pt);

  const ElemType &at(const Point<N> &pt) const;

  ElemType knn_value(const Point<N> &key, size_t k) const;

  std::vector<ElemType> knn_query(const Point<N> &key, size_t k) const;

private:
  ElemType def;
  size_t dimension_;
  size_t size_;

  struct KDTreeNode
  {
    Point<N> pt_;
    ElemType value_;
    KDTreeNode *children[2];

    KDTreeNode(const Point<N> &pt, const ElemType &value)
    {

      children[0] = nullptr;
      children[1] = nullptr;
      pt_ = pt;
      value_ = value;
    }

    KDTreeNode(Point<N> pt, ElemType value, KDTreeNode *cl, KDTreeNode *cr)
    {
      pt_ = pt;
      value_ = value;
      children[0] = cl;
      children[1] = cr;
    }
    ~KDTreeNode()
    {

    }
  };
  mutable KDTreeNode *root;

  bool search(const Point<N> &pt, KDTreeNode **&result)
  {
    int dimension{0};
    result = &root;

    while (*result && (*result)->pt_ != pt)
    {
      dimension = dimension > dimension_ - 1 ? 0 : dimension;
      result = &((*result)->children[pt[dimension] > (*result)->pt_[dimension]]);
      dimension++;
    }

    return (*result) != nullptr;
  }
  bool search(const Point<N> &pt, KDTreeNode **&result) const
  {
    int dimension{0};

    result = &root;
    while (*result && (*result)->pt_ != pt)
    {
      dimension = dimension > dimension_ - 1 ? 0 : dimension;
      result = &((*result)->children[pt[dimension] > (*result)->pt_[dimension]]);
      dimension++;
    }
    return (*result) != nullptr;
  }

  void destroy(KDTreeNode *node)
  {
    if (node != nullptr)
    {
      destroy(node->children[0]);
      destroy(node->children[1]);
      delete node;
    }
  }
  static KDTreeNode *copyNodes(const KDTreeNode *node)
  {
    if (node != nullptr)
    {
      KDTreeNode *nodeCopy = new KDTreeNode(node->pt_, node->value_, copyNodes(node->children[0]), copyNodes(node->children[1]));
      return nodeCopy;
    }
  }
  void knn_util(Point<N> key, KDTreeNode *currentNode, KDTreeNode *&guest, double &bestDistance, int depth) const;
};

template <size_t N, typename ElemType>
void KDTree<N, ElemType>::knn_util(Point<N> key, KDTreeNode *currentNode, KDTreeNode *&guest, double &bestDistance, int depth)
    const
{

  if (currentNode == nullptr)
    return;
  double currentDistance = distance(currentNode->pt_, key);

  if (currentDistance <= bestDistance)
  {
    bestDistance = currentDistance;
    guest = currentNode;
  }
  int axis = depth % dimension_;
  bool child = key[axis] <= currentNode->pt_[axis];

  knn_util(key, currentNode->children[!child], guest, bestDistance, ++depth);

  if (fabs(currentNode->pt_[axis] - key[axis]) <= guest->pt_[axis])
    knn_util(key, currentNode->children[child], guest, bestDistance, ++depth);
}

template <size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree()
{
  size_ = 0;
  root = nullptr;
  dimension_ = N;
  def = 0;
}

template <size_t N, typename ElemType>
KDTree<N, ElemType>::~KDTree()
{
  destroy(root);
}

template <size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree(const KDTree &rhs)
{
  root = copyNodes(rhs.root);
  dimension_ = rhs.dimension_;
  size_ = rhs.size_;
  def = rhs.def;
}

template <size_t N, typename ElemType>
KDTree<N, ElemType> &KDTree<N, ElemType>::operator=(const KDTree &rhs)
{
  root = copyNodes(rhs.root);
  dimension_ = rhs.dimension_;
  size_ = rhs.size_;
  def = rhs.def;
  return *this;
}

template <size_t N, typename ElemType>
size_t KDTree<N, ElemType>::dimension() const
{
  return dimension_;
}

template <size_t N, typename ElemType>
size_t KDTree<N, ElemType>::size() const
{
  return size_;
}

template <size_t N, typename ElemType>
bool KDTree<N, ElemType>::empty() const
{
  return size_ == 0;
}

template <size_t N, typename ElemType>
bool KDTree<N, ElemType>::contains(const Point<N> &pt) const
{
  KDTreeNode **tmp;
  bool result = search(pt, tmp);
  return result;
}

template <size_t N, typename ElemType>
void KDTree<N, ElemType>::insert(const Point<N> &pt, const ElemType &value)
{
  KDTreeNode **tmp;
  if (empty())
  {
    ++size_;
    this->root = new KDTreeNode(pt, value);
  }
  else if (search(pt, tmp))
  {
    at(pt) = value;
  }
  else
  {
    ++size_;
    (*tmp) = new KDTreeNode(pt, value);
  }
}

template <size_t N, typename ElemType>
ElemType &KDTree<N, ElemType>::operator[](const Point<N> &pt)
{
  KDTreeNode **tmp;
  if (search(pt, tmp))
    return (*tmp)->value_;
  insert(pt, def);
  search(pt, tmp);
  return (*tmp)->value_;
  ;
}

template <size_t N, typename ElemType>
ElemType &KDTree<N, ElemType>::at(const Point<N> &pt)
{
  KDTreeNode **tmp;
  if (!search(pt, tmp))
  {
    throw std::out_of_range("");
  }
  return ((*tmp)->value_);
}

template <size_t N, typename ElemType>
const ElemType &KDTree<N, ElemType>::at(const Point<N> &pt) const
{
  KDTreeNode **tmp;
  if (!search(pt, tmp))
  {
    throw std::out_of_range("");
  }
  return ((*tmp)->value_);
}

template <size_t N, typename ElemType>
ElemType KDTree<N, ElemType>::knn_value(const Point<N> &key, size_t k) const
{
  KDTreeNode *currentNode = root;
  KDTreeNode *guest = nullptr;
  double bestDistance = std::numeric_limits<double>::infinity();
  size_t depth{0};

  knn_util(key, currentNode, guest, bestDistance, depth);
  return guest->value_;
}

template <size_t N, typename ElemType>
std::vector<ElemType> KDTree<N, ElemType>::knn_query(const Point<N> &key, size_t k) const
{
  KDTreeNode *currentNode = root;
  KDTreeNode *guest = nullptr;
  double bestDistance = std::numeric_limits<double>::infinity();
  size_t depth{0};

  knn_util(key, currentNode, guest, bestDistance, depth);
  return guest->value_;
}

#endif // SRC_KDTREE_HPP_
