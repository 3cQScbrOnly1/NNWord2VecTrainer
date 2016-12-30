
include_directories(
  model
)

add_executable(NNWordEmbTrainer NNWordEmbTrainer.cpp)
target_link_libraries(NNWordEmbTrainer ${LIBS})
