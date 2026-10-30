add_library(hybrid-state STATIC hybrid-state.cpp)
target_link_libraries(hybrid-state PUBLIC llama vendor::hash vendor::nlohmann)
target_include_directories(hybrid-state PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)
target_compile_features(hybrid-state PUBLIC cxx_std_17)

add_executable(llama-hybrid-prefill prefill.cpp)
target_link_libraries(llama-hybrid-prefill PRIVATE hybrid-state vendor::cpp-httplib)
target_include_directories(llama-hybrid-prefill PRIVATE ${PROJECT_SOURCE_DIR}/vendor/cpp-httplib)

add_executable(llama-hybrid-check check.cpp)
target_link_libraries(llama-hybrid-check PRIVATE hybrid-state)
