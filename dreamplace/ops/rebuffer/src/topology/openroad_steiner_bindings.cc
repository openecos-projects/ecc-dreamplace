// SPDX-License-Identifier: BSD-3-Clause
// Python boundary for the OpenROAD-compatible placement Steiner provider.

#include <stdexcept>
#include <string>
#include <memory>
#include <mutex>
#include <vector>

#include "openroad_steiner.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace {

std::mutex& fluteMutex()
{
  static std::mutex mutex;
  return mutex;
}

std::unique_ptr<rebuffer::openroad::flt::Flute>& cachedFlute()
{
  static std::unique_ptr<rebuffer::openroad::flt::Flute> flute;
  return flute;
}

std::string& cachedPostLut()
{
  static std::string path;
  return path;
}

std::string& cachedPowvLut()
{
  static std::string path;
  return path;
}

rebuffer::openroad::Tree fluteTree(const std::vector<int>& x,
                                   const std::vector<int>& y,
                                   int accuracy,
                                   const std::string& post_lut,
                                   const std::string& powv_lut)
{
  std::lock_guard<std::mutex> lock(fluteMutex());
  auto& flute = cachedFlute();
  if (!flute || cachedPostLut() != post_lut || cachedPowvLut() != powv_lut) {
    flute = std::make_unique<rebuffer::openroad::flt::Flute>();
    flute->setLutPaths(post_lut, powv_lut);
    cachedPostLut() = post_lut;
    cachedPowvLut() = powv_lut;
  }
  return flute->flute(x, y, accuracy);
}

py::dict treeToDict(const rebuffer::openroad::Tree& tree,
                    const std::string& provider)
{
  py::list branch_x;
  py::list branch_y;
  py::list branch_n;
  py::list edges;
  for (int index = 0; index < tree.branchCount(); ++index) {
    const auto& branch = tree.branch[index];
    branch_x.append(branch.x);
    branch_y.append(branch.y);
    branch_n.append(branch.n);
    if (branch.n >= 0 && branch.n < tree.branchCount()) {
      const auto& neighbor = tree.branch[branch.n];
      if (branch.x != neighbor.x || branch.y != neighbor.y) {
        edges.append(py::make_tuple(branch.x, branch.y, neighbor.x, neighbor.y));
      }
    }
  }

  py::dict result;
  result["status"] = "ok";
  result["provider"] = provider;
  result["deg"] = tree.deg;
  result["wirelength"] = tree.length;
  result["branch_x"] = std::move(branch_x);
  result["branch_y"] = std::move(branch_y);
  result["branch_n"] = std::move(branch_n);
  result["edges"] = std::move(edges);
  result["branch_count"] = tree.branchCount();
  return result;
}

rebuffer::openroad::Tree makeSttTree(const std::vector<int>& x,
                                     const std::vector<int>& y,
                                     int driver_index,
                                     float alpha,
                                     const std::string& post_lut,
                                     const std::string& powv_lut,
                                     bool& used_pd)
{
  used_pd = false;
  if (alpha > 0.0f) {
    auto tree
        = rebuffer::openroad::pdr::primDijkstra(x, y, driver_index, alpha);
    if (rebuffer::openroad::checkTree(tree)) {
      used_pd = true;
      return tree;
    }
  }

  return fluteTree(x, y, 3, post_lut, powv_lut);
}

}  // namespace

PYBIND11_MODULE(rebuffer_steiner_cpp, module)
{
  module.doc()
      = "OpenROAD-compatible Prim-Dijkstra and FLUTE placement Steiner provider";

  module.def(
      "prim_dijkstra",
      [](const std::vector<int>& x,
         const std::vector<int>& y,
         int driver_index,
         float alpha) {
        return treeToDict(
            rebuffer::openroad::pdr::primDijkstra(
                x, y, driver_index, alpha),
            "openroad_prim_dijkstra");
      },
      py::arg("x"),
      py::arg("y"),
      py::arg("driver_index"),
      py::arg("alpha") = 0.3f);

  module.def(
      "flute",
      [](const std::vector<int>& x,
         const std::vector<int>& y,
         int accuracy,
         const std::string& post_lut,
         const std::string& powv_lut) {
        return treeToDict(
            fluteTree(x, y, accuracy, post_lut, powv_lut), "openroad_flute");
      },
      py::arg("x"),
      py::arg("y"),
      py::arg("accuracy") = 3,
      py::arg("post_lut"),
      py::arg("powv_lut"));

  module.def(
      "stt",
      [](const std::vector<int>& x,
         const std::vector<int>& y,
         int driver_index,
         float alpha,
         const std::string& post_lut,
         const std::string& powv_lut) {
        bool used_pd = false;
        auto tree = makeSttTree(
            x, y, driver_index, alpha, post_lut, powv_lut, used_pd);
        return treeToDict(
            tree, used_pd ? "openroad_prim_dijkstra" : "openroad_flute");
      },
      py::arg("x"),
      py::arg("y"),
      py::arg("driver_index"),
      py::arg("alpha") = 0.3f,
      py::arg("post_lut"),
      py::arg("powv_lut"));
}
