#change_here
set(SUBGRAPH_MATCHING_FILE_LIST
        KF/cs.h
        KF/spectra.cpp
        KF/spectra.h
        matchingcommand.cpp
        matchingcommand.h
        FilterVertices.cpp
        FilterVertices.h
        IO.cpp
        IO.h
        eigenHelper.cpp
        eigenHelper.h
        BuildTable.cpp
        BuildTable.h
        GenerateQueryPlan.cpp
        GenerateQueryPlan.h
        EvaluateQuery.cpp
        EvaluateQuery.h
        GenerateFilteringPlan.cpp
        GenerateFilteringPlan.h
        StudyPerformance.cpp)

set(GRAPH_CONVERTER_FILE_LIST
        GraphConverter.cpp)

add_executable(SubgraphMatching.out ${SUBGRAPH_MATCHING_FILE_LIST})
target_link_libraries(SubgraphMatching.out graph utility nd et)

add_executable(GraphConverter.out ${GRAPH_CONVERTER_FILE_LIST})
target_link_libraries(GraphConverter.out graph utility)

add_executable(EdgeListConverter.out EdgeListToCSR.cpp)
add_executable(ReassignLabel ReassignLabel.cpp)
add_executable(SelectEdge SelectEdge.cpp)