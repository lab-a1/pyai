#include "../../cppai/knn.hpp"
#include <gtest/gtest.h>

TEST(kNNTest, Example1) {
    kNN knn;
    knn.train();

    EXPECT_STRNE("hello", "world");
}
