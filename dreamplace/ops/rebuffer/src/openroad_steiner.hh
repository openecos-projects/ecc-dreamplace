// SPDX-License-Identifier: BSD-3-Clause
// Derived from OpenROAD src/stt (Copyright (c) 2019-2025, The OpenROAD
// Authors).  The graph and LUT loading are reduced to the subset needed by
// the ECC rebuffer provider.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "boost/multi_array.hpp"

namespace rebuffer::openroad {

struct Branch
{
  int x;
  int y;
  int n;
};

struct Tree
{
  int deg;
  int length;
  std::vector<Branch> branch;

  int branchCount() const { return static_cast<int>(branch.size()); }
};

bool checkTree(const Tree& tree);

namespace pdr {

Tree primDijkstra(const std::vector<int>& x,
                  const std::vector<int>& y,
                  int driver_index,
                  float alpha);

}  // namespace pdr

namespace flt {

class Flute
{
 public:
  Flute() = default;
  ~Flute();

  void setLutPaths(std::string post_path, std::string powv_path);
  Tree flute(const std::vector<int>& x,
             const std::vector<int>& y,
             int acc);
  Tree flutes(const std::vector<int>& xs,
              const std::vector<int>& ys,
              const std::vector<int>& s,
              int acc);
  int flute_wl(int d,
               const std::vector<int>& x,
               const std::vector<int>& y,
               int acc);
  int wirelength(const Tree& tree);
  void plottree(const Tree& tree);

 private:
  struct Csoln;

  using LutType = boost::multi_array<std::shared_ptr<Csoln[]>, 2>*;
  using NumSoln = int**;

  void readLUT();
  void makeLUT(LutType& lut, NumSoln& numsoln);
  void initLUT(int to_d, LutType lut, NumSoln numsoln);
  void ensureLUT(int d);
  void deleteLUT();
  void deleteLUT(LutType& lut, NumSoln& numsoln);

  Tree d_merge_tree(const Tree& t1, const Tree& t2) const;
  Tree h_merge_tree(const Tree& t1,
                    const Tree& t2,
                    const std::vector<int>& s);
  Tree v_merge_tree(const Tree& t1, const Tree& t2) const;
  void local_refinement(int deg, Tree* tp, int p);

  Tree flutes_low_degree(int d,
                         const std::vector<int>& xs,
                         const std::vector<int>& ys,
                         const std::vector<int>& s);
  Tree flutes_medium_degree(int d,
                            const std::vector<int>& xs,
                            const std::vector<int>& ys,
                            const std::vector<int>& s,
                            int acc);
  int flutes_wl_low_degree(int d,
                           const std::vector<int>& xs,
                           const std::vector<int>& ys,
                           const std::vector<int>& s);
  int flutes_wl_medium_degree(int d,
                              const std::vector<int>& xs,
                              const std::vector<int>& ys,
                              const std::vector<int>& s,
                              int acc);
  int flutes_wl_all_degree(int d,
                           const std::vector<int>& xs,
                           const std::vector<int>& ys,
                           const std::vector<int>& s,
                           int acc);
  Tree flutes_all_degree(int d,
                         const std::vector<int>& xs,
                         const std::vector<int>& ys,
                         const std::vector<int>& s,
                         int acc);

  LutType lut_ = nullptr;
  NumSoln numsoln_ = nullptr;
  int lut_valid_d_ = 0;

  std::string post_path_;
  std::string powv_path_;
  std::string powv_data_;
  std::string post_data_;

  static constexpr int kLutInitialDegree = 8;
  static constexpr int kNumGroup[10]
      = {0, 0, 0, 0, 6, 30, 180, 1260, 10080, 90720};
};

}  // namespace flt

}  // namespace rebuffer::openroad
