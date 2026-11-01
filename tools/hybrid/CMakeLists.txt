add_library(hybrid-state STATIC hybrid-state.cpp)
set_target_properties(hybrid-state PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hybrid-state PUBLIC llama vendor::hash vendor::nlohmann)
target_include_directories(hybrid-state PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)
target_compile_features(hybrid-state PUBLIC cxx_std_17)

add_executable(llama-hybrid-prefill prefill.cpp)
target_link_libraries(llama-hybrid-prefill PRIVATE hybrid-state vendor::cpp-httplib)
target_include_directories(llama-hybrid-prefill PRIVATE ${PROJECT_SOURCE_DIR}/vendor/cpp-httplib)

add_executable(llama-hybrid-check check.cpp)
target_link_libraries(llama-hybrid-check PRIVATE hybrid-state)

add_executable(llama-hybrid-check-append check-append.cpp)
target_link_libraries(llama-hybrid-check-append PRIVATE hybrid-state)

add_executable(llama-hybrid-check-guards check-guards.cpp)
target_link_libraries(llama-hybrid-check-guards PRIVATE hybrid-state)

add_executable(llama-hybrid-check-distributions check-distributions.cpp)
target_link_libraries(llama-hybrid-check-distributions PRIVATE hybrid-state)

add_executable(llama-hybrid-check-client check-client.cpp)
target_link_libraries(llama-hybrid-check-client PRIVATE hybrid-state)

add_executable(llama-hybrid-check-codec check-codec.cpp)
target_link_libraries(llama-hybrid-check-codec PRIVATE hybrid-state)

add_executable(llama-hybrid-bench-check ../hybrid-bench/check.cpp)
target_link_libraries(llama-hybrid-bench-check PRIVATE hybrid-state)

add_executable(llama-hybrid-check-identity check-identity.cpp)
target_link_libraries(llama-hybrid-check-identity PRIVATE hybrid-state)

add_executable(llama-hybrid-score ../hybrid-bench/score.cpp)
target_link_libraries(llama-hybrid-score PRIVATE hybrid-state)
