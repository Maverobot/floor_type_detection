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
using mlpack::optimization::Adam;
using mlpack::optimization::RMSProp;

std::tuple<arma::mat, arma::mat>
getTrainingData(double i_max = 100000, double x_min = -5, double x_max = 5) {
  arma::mat x(i_max, 1);
  arma::mat y(i_max, 1);

  size_t i = 0;
  while (i < i_max) {
    x.at(i) = x_min + (x_max - x_min) * i / i_max;
    y.at(i) = std::cos(x.at(i));
    i++;
  }

  return std::make_tuple(x, y);
};

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

  arma::mat train_x;
  arma::mat train_y;

  std::tie(train_x, train_y) = getTrainingData(std::stod(argv[1]));
  std::cout << "training data size: " << train_x.size() << std::endl;
  // std::cout << "training x: " << std::endl << train_x << std::endl;
  // std::cout << "training y: " << std::endl << train_y << std::endl;

  FFN<MeanSquaredError<>> model;
  model.Add<Linear<>>(1, 10);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(10, 40);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(40, 40);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(40, 1);
  model.ResetParameters();

  RMSProp opt(0.01, 64, 0.99, 1e-8, 1000000, 1e-5, true);
  model.Train(train_x, train_y, opt);

  arma::mat test_x(1, 1);
  test_x.at(0) = std::stod(argv[2]);
  arma::mat test_y(1, 1);
  model.Predict(test_x, test_y);

  std::cout << "predicted: " << test_y << "real: " << std::cos(test_x(0))
            << std::endl;

  return 0;
}
