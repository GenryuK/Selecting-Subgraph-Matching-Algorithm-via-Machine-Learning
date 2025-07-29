



#ifndef SUBGRAPHMATCHING_GRAPH_H
#define SUBGRAPHMATCHING_GRAPH_H

#include <unordered_map>
#include <iostream>
#include <vector>
#include "utility/sparsepp/spp.h"
#include "configuration/types.h"
#include "configuration/config.h"


using spp::sparse_hash_map;
class Graph {
private:
    bool enable_label_offset_;

    ui vertices_count_;
    ui edges_count_;
    ui labels_count_;
    ui max_degree_;
    ui max_label_frequency_;

    ui* offsets_;
    VertexID * neighbors_;
    LabelID* labels_;
    ui* reverse_index_offsets_;
    ui* reverse_index_;

    int* core_table_;
    ui core_length_;

    std::unordered_map<LabelID, ui> labels_frequency_;
    sparse_hash_map<uint64_t, std::vector<edge>* >* edge_index_;

#if OPTIMIZED_LABELED_GRAPH == 1
    ui* labels_offsets_;
    std::unordered_map<LabelID, ui>* nlf_;
#endif

private:
    void BuildReverseIndex();

#if OPTIMIZED_LABELED_GRAPH == 1
    void BuildNLF();
    //void BuildLabelOffset();
#endif

public:
    void BuildLabelOffset();
    Graph(const bool enable_label_offset) {
        enable_label_offset_ = enable_label_offset;

        vertices_count_ = 0;
        edges_count_ = 0;
        labels_count_ = 0;
        max_degree_ = 0;
        max_label_frequency_ = 0;
        core_length_ = 0;

        offsets_ = NULL;
        neighbors_ = NULL;
        labels_ = NULL;
        reverse_index_offsets_ = NULL;
        reverse_index_ = NULL;
        core_table_ = NULL;
        labels_frequency_.clear();
        edge_index_ = NULL;
#if OPTIMIZED_LABELED_GRAPH == 1
        //void BuildLabelOffset();
        labels_offsets_ = NULL;
        nlf_ = NULL;
#endif
    }

    ~Graph() {
        delete[] offsets_;
        delete[] neighbors_;
        delete[] labels_;
        delete[] reverse_index_offsets_;
        delete[] reverse_index_;
        delete[] core_table_;
        delete edge_index_;
#if OPTIMIZED_LABELED_GRAPH == 1
        delete[] labels_offsets_;
        delete[] nlf_;
#endif
    }

public:
    void loadGraphFromFile(const std::string& file_path);
    void loadGraphFromFileCompressed(const std::string& degree_path, const std::string& edge_path,
                                     const std::string& label_path);
    void storeComparessedGraph(const std::string& degree_path, const std::string& edge_path,
                               const std::string& label_path);
    void printGraphMetaData();
public:
    void getNeighborsByLabelCount(const VertexID id, const LabelID label, ui &count) const
    {
        ui offset = id * labels_count_ + label;
        count = labels_offsets_[offset + 1] - labels_offsets_[offset];
    }
    const ui getLabelsCount() const {
        return labels_count_;
    }

    const ui getVerticesCount() const {
        return vertices_count_;
    }

    const ui getEdgesCount() const {
        return edges_count_;
    }

    const ui getGraphMaxDegree() const {
        return max_degree_;
    }

    const ui getGraphMaxLabelFrequency() const {
        return max_label_frequency_;
    }

    const ui getVertexDegree(const VertexID id) const {
        return offsets_[id + 1] - offsets_[id];
    }

    const ui getLabelsFrequency(const LabelID label) const {
        return labels_frequency_.find(label) == labels_frequency_.end() ? 0 : labels_frequency_.at(label);
    }

    const ui getCoreValue(const VertexID id) const {
        return core_table_[id];
    }

    const ui get2CoreSize() const {
        return core_length_;
    }
    const LabelID getVertexLabel(const VertexID id) const {
        return labels_[id];
    }

    const ui * getVertexNeighbors(const VertexID id, ui& count) const {
        count = offsets_[id + 1] - offsets_[id];
        return neighbors_ + offsets_[id];
    }

    const sparse_hash_map<uint64_t, std::vector<edge>*>* getEdgeIndex() const {
        return edge_index_;
    }

    const ui * getVerticesByLabel(const LabelID id, ui& count) const {
        count = reverse_index_offsets_[id + 1] - reverse_index_offsets_[id];
        return reverse_index_ + reverse_index_offsets_[id];
    }

#if OPTIMIZED_LABELED_GRAPH == 1
    const ui * getNeighborsByLabel(const VertexID id, const LabelID label, ui& count) const {
        ui offset = id * labels_count_ + label;
        count = labels_offsets_[offset + 1] - labels_offsets_[offset];
        return neighbors_ + labels_offsets_[offset];
    }

    const std::unordered_map<LabelID, ui>* getVertexNLF(const VertexID id) const {
        return nlf_ + id;
    }

    bool checkEdgeExistence(const VertexID u, const VertexID v, const LabelID u_label) const {
        ui count = 0;
        const VertexID* neighbors = getNeighborsByLabel(v, u_label, count);
        int begin = 0;
        int end = count - 1;
        while (begin <= end) {
            int mid = begin + ((end - begin) >> 1);
            if (neighbors[mid] == u) {
                return true;
            }
            else if (neighbors[mid] > u)
                end = mid - 1;
            else
                begin = mid + 1;
        }

        return false;
    }
#endif

    bool checkEdgeExistence(VertexID u, VertexID v) const {
        //デバッグのためこちらを一旦コメントアウト
        // if (getVertexDegree(u) < getVertexDegree(v)) {
        //     std::swap(u, v);
        // }
        // ui count = 0;
        // const VertexID* neighbors =  getVertexNeighbors(v, count);

        // int begin = 0;
        // int end = count - 1;
        // while (begin <= end) {
        //     int mid = begin + ((end - begin) >> 1);
        //     if (neighbors[mid] == u) {
        //         return true;
        //     }
        //     else if (neighbors[mid] > u)
        //         end = mid - 1;
        //     else
        //         begin = mid + 1;
        // }

        // return false;

        // 特定の頂点の組み合わせの時だけデバッグ情報を出力する
        bool enable_debug = ((u == 2618 && v == 622) || (u == 622 && v == 2618));

        if (enable_debug) {
            std::cout << "\n// =================== DEBUG START: checkEdgeExistence (2 args) =================== \\\\" << std::endl;
            std::cout << "|| Initial call -> u: " << u << " (degree: " << getVertexDegree(u) << "), v: " << v << " (degree: " << getVertexDegree(v) << ")" << std::endl;
        }

        // 最適化：次数(隣接頂点の数)が少ない方の頂点の隣接リストを調べる
        if (getVertexDegree(u) < getVertexDegree(v)) {
            if (enable_debug) {
                std::cout << "|| -> Swapping u and v because degree(" << u << ") < degree(" << v << ")" << std::endl;
            }
            std::swap(u, v);
            if (enable_debug) {
                std::cout << "||    After swap  -> u: " << u << ", v: " << v << std::endl;
            }
        }

        ui count = 0;
        // 次数が少ない方(v)の隣接リストを取得
        const VertexID* neighbors = getVertexNeighbors(v, count);

        if (enable_debug) {
            std::cout << "|| 1. getVertexNeighbors(v=" << v << ") Result:" << std::endl;
            std::cout << "||    - Neighbor Count: " << count << std::endl;
            std::cout << "||    - Neighbor List: [ ";
            for (ui i = 0; i < count; ++i) {
                std::cout << neighbors[i] << " ";
            }
            std::cout << "]" << std::endl;
            std::cout << "|| 2. Starting Binary Search to find '" << u << "' in the list..." << std::endl;
        }

        int begin = 0;
        int end = count - 1;
        while (begin <= end) {
            int mid = begin + ((end - begin) >> 1);
            
            if (enable_debug) {
                std::cout << "||    - Loop: begin=" << begin << ", end=" << end << ", mid=" << mid << " -> neighbors[mid]=" << neighbors[mid] << std::endl;
            }

            if (neighbors[mid] == u) {
                if (enable_debug) {
                    std::cout << "||    -> Found " << u << " at index " << mid << ". Returning true." << std::endl;
                    std::cout << "\\\\ ===================  DEBUG END: Edge found  =================== //" << std::endl;
                }
                return true;
            }
            else if (neighbors[mid] > u)
                end = mid - 1;
            else
                begin = mid + 1;
        }

        if (enable_debug) {
            std::cout << "||    -> Loop finished. " << u << " was not found. Returning false." << std::endl;
            std::cout << "\\\\ =================== DEBUG END: Edge NOT found =================== //" << std::endl;
        }

        return false;
    }

    void buildCoreTable();

    void buildEdgeIndex();
};


#endif 
