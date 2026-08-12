/*************************************************************************
    > File Name: Util.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Jun 2015 04:08:18 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_UTILITY_UTIL_H
#define DREAMPLACE_UTILITY_UTIL_H

#include "utility/src/namespace.h"

DREAMPLACE_BEGIN_NAMESPACE

enum Direction1DType {
  kLOW = 0,
  kHIGH = 1,
  kX = 0,
  kY = 1,
  kLEFT = 0,
  kRIGHT = 1,
  kBOTTOM = 0,
  kTOP = 1
};

enum Direction2DType { kXLOW = 0, kXHIGH = 1, kYLOW = 2, kYHIGH = 3 };

inline Direction1DType getXY(Direction2DType d) {
  return Direction1DType(d > 1);
}

inline Direction1DType getLH(Direction2DType d) {
  return Direction1DType(d & 1);
}

inline Direction2DType to2D(Direction1DType xy, Direction1DType lh) {
  return Direction2DType((static_cast<int>(xy) << 1) + static_cast<int>(lh));
}

template <typename T>
struct coordinate_traits;

template <>
struct coordinate_traits<int> {
  typedef int coordinate_type;
  typedef double euclidean_distance_type;
  typedef long manhattan_distance_type;
  typedef long area_type;
  typedef unsigned int site_index_type;
  typedef unsigned long site_area_type;
  typedef unsigned int index_type;
  typedef double weight_type;
};

template <>
struct coordinate_traits<unsigned int> {
  typedef unsigned int coordinate_type;
  typedef double euclidean_distance_type;
  typedef long manhattan_distance_type;
  typedef long area_type;
  typedef unsigned int site_index_type;
  typedef unsigned long site_area_type;
  typedef unsigned int index_type;
  typedef double weight_type;
};

template <>
struct coordinate_traits<float> {
  typedef float coordinate_type;
  typedef double euclidean_distance_type;
  typedef double manhattan_distance_type;
  typedef double area_type;
  typedef unsigned int site_index_type;
  typedef double site_area_type;
  typedef unsigned int index_type;
  typedef float weight_type;
};

template <>
struct coordinate_traits<double> {
  typedef double coordinate_type;
  typedef long double euclidean_distance_type;
  typedef long double manhattan_distance_type;
  typedef long double area_type;
  typedef unsigned long site_index_type;
  typedef long double site_area_type;
  typedef unsigned long index_type;
  typedef double weight_type;
};

DREAMPLACE_END_NAMESPACE

#endif
