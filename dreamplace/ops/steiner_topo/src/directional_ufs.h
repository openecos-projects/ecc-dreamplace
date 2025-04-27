// directional_ufs.h
#ifndef DIRECTIONAL_UFS_H
#define DIRECTIONAL_UFS_H

#include "place_io/src/Point.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>
#include <map>
#include <set>

DREAMPLACE_BEGIN_NAMESPACE

template<typename CoordType> class BasicUFS {
  public:
    std::vector<int> parent;
    std::vector<CoordType> distance;
  public:
    BasicUFS(int n) : parent(n), distance(n, CoordType{}) {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }
    
    BasicUFS() = default;

    CoordType dist(int u) { return distance[u]; }

    int findfa(int u) {
        if (parent[u] == u) return u;
        int root = findfa(parent[u]);
        distance[u] += distance[parent[u]];
        return parent[u] = root;
    }

    void unite(int u, int v, CoordType dis) {
        int ru = findfa(u);
        int rv = findfa(v);
        parent[rv] = ru;
        distance[rv] = dis + dist(u) - dist(v);
    }
};

template <typename CoordType> class UnifiedUFS {
  public:
    enum Direction : int {
        DOWN = 0, UP = 1, LEFT = 2, RIGHT = 3, COUNT = 4
    };
  private:
    std::vector<BasicUFS<CoordType>> ufs;
    int vertex_count;

  public:
    UnifiedUFS(int n) : ufs(Direction::COUNT, BasicUFS<CoordType>(n)), vertex_count(n) {}

    // connect v to u
    void unite(int u, int v, Point<CoordType> pos_u, Point<CoordType> pos_v) {
        if (pos_u.y() == pos_v.y()) {
            // Horizontal connection
            CoordType dx = pos_v.x() - pos_u.x();
             if (dx > 0) {
                 ufs[LEFT].unite(u, v, dx);
             } else if (dx < 0) {
                 ufs[RIGHT].unite(u, v, dx); 
             }
        } else if (pos_u.x() == pos_v.x()) {
            // Vertical connection
            CoordType dy = pos_v.y() - pos_u.y();
             if (dy > 0) {
                 ufs[DOWN].unite(u, v, dy);
             } else if (dy < 0) {
                 ufs[UP].unite(u, v, dy); 
             }
        }
    }
    std::pair<int, int> getRelateVertex(int u, int limit, std::map<int, 
                                        std::vector<int>> &steiner_adj_vertices_map, 
                                        std::vector<int> &newx, std::vector<int> &newy) {
        std::vector<int> relate_pins(2, -1);
        std::vector<int> root(Direction::COUNT);
        for (int direction = 0; direction < Direction::COUNT; ++direction) {
            root[direction] = ufs[direction].findfa(u);
        }

        // operate in direction pairs:
        // DOWN-UP, LEFT-RIGHT
        for (int dir_small = 0; dir_small < Direction::COUNT; dir_small += 2) {
            int dir_large = dir_small + 1;
            int cur_idx = dir_small / 2;

            if (root[dir_small] < limit) {
                relate_pins[cur_idx] = root[dir_small];
            }
            if (root[dir_large] < limit) {
                if (relate_pins[cur_idx] == -1) {
                    relate_pins[cur_idx] = root[dir_large];
                } else {
                    relate_pins[cur_idx] = ufs[dir_small].dist(u) < ufs[dir_large].dist(u) ? root[dir_small] : root[dir_large];
                }
            }

            // If the current index is not assigned a pin in its line, we need to find the closest pin
            if (relate_pins[cur_idx] == -1) {
                std::set<int> candidate_pins;
                for (auto vertex : steiner_adj_vertices_map[root[dir_small]]) {
                    candidate_pins.insert(ufs[dir_small].findfa(vertex));
                }
                for (auto vertex : steiner_adj_vertices_map[root[dir_large]]) {
                    candidate_pins.insert(ufs[dir_large].findfa(vertex));
                }
                assert(candidate_pins.size() < 4);
                auto min_pin = std::min_element(candidate_pins.begin(), candidate_pins.end(), 
                    [dir_small, &newx, &newy, u](const int& a, const int& b) {
                        return (dir_small == LEFT) ? 
                                std::abs(newy[u] - newy[a]) < std::abs(newy[u] - newy[b])
                                : std::abs(newx[u] - newx[a]) < std::abs(newx[u] - newx[b]);
                    });
                relate_pins[cur_idx] = *min_pin;
            }
        }
        return std::make_pair(relate_pins[0], relate_pins[1]);
    }
};

DREAMPLACE_END_NAMESPACE

#endif // DIRECTIONAL_UFS_H