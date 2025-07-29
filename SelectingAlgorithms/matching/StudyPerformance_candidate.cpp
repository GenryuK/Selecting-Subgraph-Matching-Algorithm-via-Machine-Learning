



#include <chrono>
#include <future>
#include <thread>
#include <fstream>

#include "matchingcommand.h"
#include "graph/graph.h"
#include "GenerateFilteringPlan.h"
#include "FilterVertices.h"
#include "BuildTable.h"
#include "GenerateQueryPlan.h"
#include "EvaluateQuery.h"

#include "utility/analyze_symmetry/analyze_symmetry.h"
//change here
#include "KF/spectra.h"
#include <ctime>
#define NANOSECTOSEC(elapsed_time) ((elapsed_time)/(double)1000000000)
#define BYTESTOMB(memory_cost) ((memory_cost)/(double)(1024 * 1024))

size_t enumerate(Graph* data_graph, Graph* query_graph, Edges*** edge_matrix, ui** candidates, ui* candidates_count,
                ui* matching_order, size_t output_limit) {
    static ui order_id = 0;

    order_id += 1;

    auto start = std::chrono::high_resolution_clock::now();
    size_t call_count = 0;
    size_t valid_vtx_count = 0;
    size_t embedding_count = EvaluateQuery::LFTJ(data_graph, query_graph, edge_matrix, candidates, candidates_count,
                               matching_order, output_limit, call_count, valid_vtx_count);

    auto end = std::chrono::high_resolution_clock::now();
    double enumeration_time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#ifdef SPECTRUM
    if (EvaluateQuery::exit_) {
        printf("Spectrum Order %u status: Timeout\n", order_id);
    }
    else {
        printf("Spectrum Order %u status: Complete\n", order_id);
    }
#endif
    printf("Spectrum Order %u Enumerate time (seconds): %.4lf\n", order_id, NANOSECTOSEC(enumeration_time_in_ns));
    printf("Spectrum Order %u #Embeddings: %zu\n", order_id, embedding_count);
    printf("Spectrum Order %u Call Count: %zu\n", order_id, call_count);
    printf("Spectrum Order %u Per Call Count Time (nanoseconds): %.4lf\n", order_id, enumeration_time_in_ns / (call_count == 0 ? 1 : call_count));

    return embedding_count;
}

void spectrum_analysis(Graph* data_graph, Graph* query_graph, Edges*** edge_matrix, ui** candidates, ui* candidates_count,
                       size_t output_limit, std::vector<std::vector<ui>>& spectrum, size_t time_limit_in_sec) {

    for (auto& order : spectrum) {
        std::cout << "----------------------------" << std::endl;
        ui* matching_order = order.data();
        GenerateQueryPlan::printSimplifiedQueryPlan(query_graph, matching_order);

        std::future<size_t> future = std::async(std::launch::async, [data_graph, query_graph, edge_matrix, candidates, candidates_count,
                                                                     matching_order, output_limit](){
            return enumerate(data_graph, query_graph, edge_matrix, candidates, candidates_count, matching_order, output_limit);
        });

        std::cout << "execute...\n";
        std::future_status status;
        do {
            status = future.wait_for(std::chrono::seconds(time_limit_in_sec));
            if (status == std::future_status::deferred) {
                std::cout << "Deferred\n";
                exit(-1);
            } else if (status == std::future_status::timeout) {
#ifdef SPECTRUM
                EvaluateQuery::exit_ = true;
#endif
            }
        } while (status != std::future_status::ready);
    }
}

int main(int argc, char** argv) {
    MatchingCommand command(argc, argv);
    std::string input_query_graph_file = command.getQueryGraphFilePath();
    std::string input_data_graph_file = command.getDataGraphFilePath();
    std::string input_filter_type = command.getFilterType();
    std::string input_order_type = command.getOrderType();
    std::string input_engine_type = command.getEngineType();
    std::string input_max_embedding_num = command.getMaximumEmbeddingNum();
    std::string input_time_limit = command.getTimeLimit();
    std::string input_order_num = command.getOrderNum();
    std::string input_distribution_file_path = command.getDistributionFilePath();
    std::string input_csr_file_path = command.getCSRFilePath();

    std::string input_enable_symmetry = command.getEnableSymmetry();

    
    std::cout << "Command Line:" << std::endl;
    std::cout << "\tData Graph CSR: " << input_csr_file_path << std::endl;
    std::cout << "\tData Graph: " << input_data_graph_file << std::endl;
    std::cout << "\tQuery Graph: " << input_query_graph_file << std::endl;
    std::cout << "\tFilter Type: " << input_filter_type << std::endl;
    std::cout << "\tOrder Type: " << input_order_type << std::endl;
    std::cout << "\tEngine Type: " << input_engine_type << std::endl;
    std::cout << "\tOutput Limit: " << input_max_embedding_num << std::endl;
    std::cout << "\tTime Limit (seconds): " << input_time_limit << std::endl;
    std::cout << "\tOrder Num: " << input_order_num << std::endl;
    std::cout << "\tDistribution File Path: " << input_distribution_file_path << std::endl;

    std::cout << "\tEnable Symmetry: " << input_enable_symmetry << std::endl;

    std::cout << "--------------------------------------------------------------------" << std::endl;

    
    std::cout << "Load graphs..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    Graph* query_graph = new Graph(true);
    query_graph->loadGraphFromFile(input_query_graph_file);
    query_graph->buildCoreTable();

    Graph* data_graph = new Graph(true);

    if (input_csr_file_path.empty()) {
        data_graph->loadGraphFromFile(input_data_graph_file);
        if (input_filter_type == "PL"){
            data_graph->BuildLabelOffset();
        }
    }
    else {
        std::string degree_file_path = input_csr_file_path + "_deg.bin";
        std::string edge_file_path = input_csr_file_path + "_adj.bin";
        std::string label_file_path = input_csr_file_path + "_label.bin";
        data_graph->loadGraphFromFileCompressed(degree_file_path, edge_file_path, label_file_path);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double load_graphs_time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "-----" << std::endl;
    std::cout << "Query Graph Meta Information" << std::endl;
    query_graph->printGraphMetaData();
    std::cout << "-----" << std::endl;
    data_graph->printGraphMetaData();

    std::cout << "--------------------------------------------------------------------" << std::endl;

    bool enable_symmetry = false;
    if (input_enable_symmetry == "1" || input_enable_symmetry == "true") {
        enable_symmetry = true;
    }

    

    std::vector<std::set<std::pair<VertexID, VertexID>>> permutations;
    std::vector<std::pair<VertexID, VertexID>> constraints;
    std::unordered_map<VertexID, std::pair<std::set<VertexID>, std::set<VertexID>>> full_constraints;
    std::unordered_map<VertexID, std::pair<std::set<VertexID>, std::set<VertexID>>> ordered_constraints;

    if (enable_symmetry) {
        std::cout << "Analyze symmetry..." << std::endl;
        
        std::unordered_map<VertexID, std::set<VertexID>> cosets = ANALYZE_SYMMETRY::analyze_symmetry(query_graph, permutations);
        

        ANALYZE_SYMMETRY::make_constraints(cosets, constraints);
        

        ANALYZE_SYMMETRY::make_full_constraints(constraints, full_constraints);
        
    }

    


    

    std::cout << "Start queries..." << std::endl;
    std::cout << "-----" << std::endl;
    std::cout << "Filter candidates..." << std::endl;

    start = std::chrono::high_resolution_clock::now();

    clock_t start_fil = clock();

    ui** candidates = NULL;
    ui* candidates_count = NULL;
    ui* tso_order = NULL;
    TreeNode* tso_tree = NULL;
    ui* cfl_order = NULL;
    TreeNode* cfl_tree = NULL;
    ui* dpiso_order = NULL;
    TreeNode* dpiso_tree = NULL;
    TreeNode* veq_tree = NULL;
    ui* veq_order = NULL;
    TreeNode* ceci_tree = NULL;
    ui* ceci_order = NULL;
    catalog* storage = NULL;
    std::vector<std::unordered_map<VertexID, std::vector<VertexID >>> TE_Candidates;
    std::vector<std::vector<std::unordered_map<VertexID, std::vector<VertexID>>>> NTE_Candidates;
    if (input_filter_type == "LDF") {
        FilterVertices::LDFFilter(data_graph, query_graph, candidates, candidates_count);
    } else if (input_filter_type == "NLF") {
        FilterVertices::NLFFilter(data_graph, query_graph, candidates, candidates_count);
    } else if (input_filter_type == "GQL") {
        FilterVertices::GQLFilter(data_graph, query_graph, candidates, candidates_count);
    } else if (input_filter_type == "TSO") {
        FilterVertices::TSOFilter(data_graph, query_graph, candidates, candidates_count, tso_order, tso_tree);
    } else if (input_filter_type == "CFL") {
        FilterVertices::CFLFilter(data_graph, query_graph, candidates, candidates_count, cfl_order, cfl_tree);
    } else if (input_filter_type == "DPiso") {
        FilterVertices::DPisoFilter(data_graph, query_graph, candidates, candidates_count, dpiso_order, dpiso_tree);
    } else if (input_filter_type == "VEQ") {
        FilterVertices::VEQFilter(data_graph, query_graph, candidates, candidates_count, veq_order, veq_tree);
    } else if (input_filter_type == "CECI") {
        FilterVertices::CECIFilter(data_graph, query_graph, candidates, candidates_count, ceci_order, ceci_tree, TE_Candidates, NTE_Candidates);
    } else if (input_filter_type == "RM") {
        FilterVertices::RMFilter(data_graph, query_graph, candidates, candidates_count, storage);
    } else if (input_filter_type == "CaLiG") {
        FilterVertices::CaLiGFilter(data_graph, query_graph, candidates, candidates_count);
    }
    //change here
    else if(input_filter_type == "PL") {
            int dsiz = data_graph->getVerticesCount();
            float **eigenVD1 = NULL;
            eigenVD1 = new float *[dsiz];
            #ifdef EIGEN_INDEX 
            for (ui i = 0; i < dsiz; ++i)
            {
                eigenVD1[i] = new float[35];
            }

            openData1(Experiments::datagraphEigenMatrix, eigenVD1);
            #endif
        int alpha1=25;
        int beta1=500;
        float **EWeight = NULL;
            float *eigenQS = new float[query_graph->getVerticesCount()];//not used
    int qsiz = query_graph->getVerticesCount();
    Edges ***edge_matrix1 = NULL;
    edge_matrix1 = new Edges **[query_graph->getVerticesCount()];
    for (ui i = 0; i < query_graph->getVerticesCount(); ++i)
    {
        edge_matrix1[i] = new Edges *[query_graph->getVerticesCount()];
    }
        SpectralMatching(query_graph->getVerticesCount(), data_graph, query_graph, 2, candidates, candidates_count, EWeight, eigenVD1, alpha1, beta1, edge_matrix1, eigenQS);
        std::cout << "end1" << std::endl;
    }
      else {
        std::cout << "The specified filter type '" << input_filter_type << "' is not supported." << std::endl;
        exit(-1);
    }

    
    ui total_candidates_count = 0;
    for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
       total_candidates_count += candidates_count[i];
    }

    int q_vertices = 0;
    q_vertices = query_graph->getVerticesCount();

    
    if (input_filter_type != "CECI")
        FilterVertices::sortCandidates(candidates, candidates_count, query_graph->getVerticesCount());

    end = std::chrono::high_resolution_clock::now();
    double filter_vertices_time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    
#ifdef OPTIMAL_CANDIDATES
    std::vector<ui> optimal_candidates_count;
    double avg_false_positive_ratio = FilterVertices::computeCandidatesFalsePositiveRatio(data_graph, query_graph, candidates,
                                                                                          candidates_count, optimal_candidates_count);
    FilterVertices::printCandidatesInfo(query_graph, candidates_count, optimal_candidates_count);
#endif
    std::cout << "-----" << std::endl;
    std::cout << "Build indices..." << std::endl;

    start = std::chrono::high_resolution_clock::now();

    
    Edges ***edge_matrix = NULL;
    if (input_filter_type != "CECI") {
        edge_matrix = new Edges **[query_graph->getVerticesCount()];
        for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
            edge_matrix[i] = new Edges *[query_graph->getVerticesCount()];
        }

        BuildTable::buildTables(data_graph, query_graph, candidates, candidates_count, edge_matrix);
    }

    end = std::chrono::high_resolution_clock::now();
    double build_table_time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    size_t memory_cost_in_bytes = 0;
    if (input_filter_type != "CECI") {
        memory_cost_in_bytes = BuildTable::computeMemoryCostInBytes(query_graph, candidates_count, edge_matrix);
        
    }
    else {
        memory_cost_in_bytes = BuildTable::computeMemoryCostInBytes(query_graph, candidates_count, ceci_order, ceci_tree,
                TE_Candidates, NTE_Candidates);
        
    }

    clock_t end_fil = clock();
    double filtering_time = 0.0;
    filtering_time = static_cast<double>(end_fil - start_fil) / CLOCKS_PER_SEC;
    std::cout << "-----" << std::endl;
    

    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "Release memories..." << std::endl;
    
    delete[] candidates_count;
    delete[] tso_order;
    delete[] tso_tree;
    delete[] cfl_order;
    delete[] cfl_tree;
    delete[] dpiso_order;
    delete[] dpiso_tree;
    delete[] ceci_order;
    delete[] ceci_tree;
    //delete[] matching_order;
    delete storage;
    for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
        delete[] candidates[i];
    }
    delete[] candidates;

    if (edge_matrix != NULL) {
        for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
            for (ui j = 0; j < query_graph->getVerticesCount(); ++j) {
                delete edge_matrix[i][j];
            }
            delete[] edge_matrix[i];
        }
        delete[] edge_matrix;
    }
    

    delete query_graph;
    delete data_graph;

    double candidate_size = 0.0;
    candidate_size = total_candidates_count/q_vertices;

    
    std::cout << "--------------------------------------------------------------------" << std::endl;
    

    printf("total candidate size: %zu\n", total_candidates_count);
    printf("Query Graph Vertices: %d\n", q_vertices);
    printf("Filtering time (seconds): %.4lf\n", filtering_time);
    printf("candidate size: %.4lf\n", candidate_size);
    // printf("Build table time (seconds): %.4lf\n", NANOSECTOSEC(build_table_time_in_ns));
    // printf("Generate query plan time (seconds): %.4lf\n", NANOSECTOSEC(generate_query_plan_time_in_ns));
    // printf("Enumerate time (seconds): %.4lf\n", NANOSECTOSEC(enumeration_time_in_ns));
    // printf("Preprocessing time (seconds): %.4lf\n", NANOSECTOSEC(preprocessing_time_in_ns));
    // printf("Total time (seconds): %.4lf\n", NANOSECTOSEC(total_time_in_ns));
    // printf("Memory cost (MB): %.4lf\n", BYTESTOMB(memory_cost_in_bytes));
    // printf("#Embeddings: %zu\n", embedding_count);
    // printf("Call Count: %zu\n", call_count);
    // printf("Per Call Count Time (nanoseconds): %.4lf\n", enumeration_time_in_ns / (call_count == 0 ? 1 : call_count));
    std::cout << "End." << std::endl;

    
    // std::fstream output;
    // output.open("/root/subgraph/test/output/survey_0408.csv", std::ios::out | std::ios::app);
    
    // output << input_query_graph_file << ",";
    // output << input_data_graph_file << ",";
    // output << input_filter_type << ",";
    // output << input_order_type << ",";
    // output << input_engine_type << ",";
    // output << NANOSECTOSEC(load_graphs_time_in_ns) << ",";
    // output << NANOSECTOSEC(filter_vertices_time_in_ns) << ",";
    // output << NANOSECTOSEC(build_table_time_in_ns) << ",";
    // output << NANOSECTOSEC(generate_query_plan_time_in_ns) << ",";
    // output << NANOSECTOSEC(enumeration_time_in_ns) << ",";
    // output << NANOSECTOSEC(preprocessing_time_in_ns) << ",";
    // output << NANOSECTOSEC(total_time_in_ns) << ",";
    // output << embedding_count << ",";
    // output << call_count << std::endl;
    
    // output.close();

    return 0;
}