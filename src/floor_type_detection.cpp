#include <cmath>
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/rnn.hpp>

using mlpack::ann::Dropout;
using mlpack::ann::FFN;
using mlpack::ann::Linear;
using mlpack::ann::LSTM;
using mlpack::ann::MeanSquaredError;
using mlpack::ann::ReLULayer;
using mlpack::ann::RNN;
using mlpack::ann::SigmoidLayer;
using mlpack::ann::TanHLayer;
using mlpack::optimization::Adam;
using mlpack::optimization::RMSProp;

std::tuple<arma::mat, arma::mat> getTrainingData(double i_max = 1000000,
                                                 double x_min = -M_PI,
                                                 double x_max = M_PI) {
  arma::mat x(1, i_max);
  arma::mat y(1, i_max);

  size_t i = 0;
  while (i < i_max) {
    x.at(i) = x_min + (x_max - x_min) * i / i_max;
    y.at(i) = std::cos(x.at(i));
    i++;
  }

  return std::make_tuple(x, y);
};

using mlpack::math::ShuffleData;

int main(int argc, char *argv[]) {

  /*
  const size_t rho = 50;
  RNN<MeanSquaredError<>> model(rho);
  model.Add<LSTM<>>(1, 50, rho);
  model.Add<Dropout<>>(0.2);
  model.Add<LSTM<>>(50, 100, rho);
  model.Add<Dropout<>>(0.2);
  model.Add<Linear<>>(100, 1);
  model.Train(arma::cube predictors, arma::cube responses,
              OptimizerType & optimizer);
  */

  arma::mat train_x_original;
  arma::mat train_y_original;

  std::tie(train_x_original, train_y_original) = getTrainingData();
  std::cout << "training data size: " << train_x_original.size() << std::endl;
  // std::cout << "training x: " << std::endl << train_x_original << std::endl;
  // std::cout << "training y: " << std::endl << train_y_original << std::endl;

  arma::mat train_x;
  arma::mat train_y;
  ShuffleData(train_x_original, train_y_original, train_x, train_y);

  FFN<MeanSquaredError<>> model;
  model.Add<Linear<>>(1, 10);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(10, 100);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(100, 1);
  model.Add<TanHLayer<>>();
  model.ResetParameters();
  RMSProp opt(0.01, 32, 0.99, 1e-8, std::stod(argv[1]), 1e-5, false);
  model.Train(train_x, train_y, opt);

  size_t i = 0;
  arma::mat assignments(train_x.size(), 1);
  std::cout << "Predicting..." << std::endl;
  model.Predict(train_x, assignments);
  std::cout << "x: " << train_x.head_cols(10) << std::endl;
  std::cout << "f(x): " << assignments.head_cols(10) << std::endl;
  auto target_labels =
      train_x.for_each([](arma::mat::elem_type &v) { v = std::cos(v); });
  std::cout << "target f(x): " << target_labels.head_cols(10) << std::endl;

  arma::mat test_x(1, 5);
  test_x(0) = 0;
  test_x(1) = M_PI;
  test_x(2) = M_PI / 2;
  test_x(3) = M_PI / 4;
  test_x(4) = 0.6166;
  arma::mat test_y;
  model.Predict(test_x, test_y);
  std::cout << "predicted: " << test_y
            << "real: " << test_x.for_each([](auto &v) { v = std::cos(v); })
            << std::endl;

  return 0;
}
