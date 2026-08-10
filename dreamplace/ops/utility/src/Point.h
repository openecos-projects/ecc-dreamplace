/*************************************************************************
    > File Name: Point.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Jun 2015 08:29:15 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_UTILITY_POINT_H
#define DREAMPLACE_UTILITY_POINT_H

#include <cmath>

#include "utility/src/Util.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
class Point {
 public:
  typedef T coordinate_type;

  Point(coordinate_type x = 0, coordinate_type y = 0) { set(x, y); }

  Point(Point const& rhs) { copy(rhs); }

  Point& operator=(Point const& rhs) {
    copy(rhs);
    return *this;
  }

  template <typename PointType>
  explicit Point(PointType const& rhs) {
    copy(rhs);
  }

  template <typename PointType>
  Point& operator=(PointType const& rhs) {
    copy(rhs);
    return *this;
  }

  Point& set(coordinate_type x, coordinate_type y) {
    m_coords[kX] = x;
    m_coords[kY] = y;
    return *this;
  }

  Point& set(Direction1DType d, coordinate_type v) {
    m_coords[d] = v;
    return *this;
  }

  coordinate_type get(Direction1DType d) const { return m_coords[d]; }
  coordinate_type x() const { return m_coords[kX]; }
  coordinate_type y() const { return m_coords[kY]; }

  bool operator==(Point const& rhs) const {
    return m_coords[kX] == rhs.m_coords[kX] &&
           m_coords[kY] == rhs.m_coords[kY];
  }

  bool operator!=(Point const& rhs) const { return !(*this == rhs); }

  bool operator<(Point const& rhs) const {
    return m_coords[kX] < rhs.m_coords[kX] ||
           (m_coords[kX] == rhs.m_coords[kX] &&
            m_coords[kY] < rhs.m_coords[kY]);
  }

  bool operator<=(Point const& rhs) const { return !(rhs < *this); }
  bool operator>(Point const& rhs) const { return rhs < *this; }
  bool operator>=(Point const& rhs) const { return !(*this < rhs); }

  Point& operator+=(Point const& rhs) {
    m_coords[kX] += rhs.get(kX);
    m_coords[kY] += rhs.get(kY);
    return *this;
  }

  Point& operator-=(Point const& rhs) {
    m_coords[kX] -= rhs.get(kX);
    m_coords[kY] -= rhs.get(kY);
    return *this;
  }

 protected:
  template <typename PointType>
  void copy(PointType const& rhs) {
    m_coords[kX] = rhs.m_coords[kX];
    m_coords[kY] = rhs.m_coords[kY];
  }

  coordinate_type m_coords[2];
};

template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type manhattanDistance(
    Point<T> const& p1, Point<T> const& p2, Direction1DType d) {
  typedef typename coordinate_traits<T>::manhattan_distance_type distance_type;
  return p1.get(d) < p2.get(d)
             ? static_cast<distance_type>(p2.get(d) - p1.get(d))
             : static_cast<distance_type>(p1.get(d) - p2.get(d));
}

template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type manhattanDistance(
    Point<T> const& p1, Point<T> const& p2) {
  return manhattanDistance(p1, p2, kX) + manhattanDistance(p1, p2, kY);
}

template <typename T>
inline typename coordinate_traits<T>::euclidean_distance_type squareDistance(
    Point<T> const& p1, Point<T> const& p2) {
  typedef typename coordinate_traits<T>::euclidean_distance_type distance_type;
  const distance_type dx = manhattanDistance(p1, p2, kX);
  const distance_type dy = manhattanDistance(p1, p2, kY);
  return dx * dx + dy * dy;
}

template <typename T>
inline typename coordinate_traits<T>::euclidean_distance_type euclideanDistance(
    Point<T> const& p1, Point<T> const& p2) {
  return std::sqrt(squareDistance(p1, p2));
}

template <typename T>
inline Point<T> operator+(Point<T> const& rhs1, Point<T> const& rhs2) {
  Point<T> p(rhs1);
  return p += rhs2;
}

template <typename T>
inline Point<T> operator-(Point<T> const& rhs1, Point<T> const& rhs2) {
  Point<T> p(rhs1);
  return p -= rhs2;
}

DREAMPLACE_END_NAMESPACE

#endif
