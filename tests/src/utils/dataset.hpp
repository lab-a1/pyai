#include <fstream>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>

xt::xtensor<double, 2> load_iris_dataset() {
    std::string csv = "data/iris.csv";
    std::ifstream in_file;
    in_file.open(csv);
    return xt::load_csv<double>(in_file);
}
