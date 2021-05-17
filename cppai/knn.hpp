#include <iostream>
#include <xtensor/xarray.hpp>

class kNN {
  public:
    void train();

  private:
    xt::xarray<double> X;
};

void kNN::train() { std::cout << "train" << std::endl; }
