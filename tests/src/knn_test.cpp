#include "../../cppai/knn.hpp"
#include "./utils/dataset.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <xtensor/xarray.hpp>

TEST(kNNTest, Example1) {
    xt::xtensor<double, 2> iris_dataset = load_iris_dataset();
    std::cout << iris_dataset[0] << std::endl;
    /* kNN knn; */
    /* knn.train(); */

    /* EXPECT_STRNE("hello", "world"); */
}
