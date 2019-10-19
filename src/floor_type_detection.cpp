#include <iostream>
#include <torch/torch.h>
#include <vector>

struct BLSTM_Model : torch::nn::Module {
  torch::nn::LSTM lstm{nullptr};
  torch::nn::LSTM reverse_lstm{nullptr};
  torch::nn::Linear linear{nullptr};

  BLSTM_Model(uint64_t layers, uint64_t hidden, uint64_t inputs,
              uint64_t outputs, uint64_t batch, uint64_t directions)
      : hidden_(hidden), batch_(batch), directions_(directions) {
    lstm = register_module(
        "lstm",
        torch::nn::LSTM(torch::nn::LSTMOptions(inputs, hidden).layers(layers)));
    reverse_lstm = register_module(
        "rlstm",
        torch::nn::LSTM(torch::nn::LSTMOptions(inputs, hidden).layers(layers)));
    linear = register_module("linear",
                             torch::nn::Linear(hidden * directions, outputs));
  }

  torch::Tensor forward(torch::Tensor x) {
    // Reverse and feed into LSTM + Reversed LSTM
    auto lstm1 = lstm->forward(x.view({x.size(0), batch_, -1}));
    //[sequence,batch,FEATURE]
    auto lstm2 =
        reverse_lstm->forward(torch::flip(x, 0).view({x.size(0), batch_, -1}));
    // Reverse Output from Reversed LSTM + Combine Outputs into one Tensor
    auto cat = torch::empty({directions_, batch_, x.size(0), hidden_});
    //[directions,batch,sequence,FEATURE]
    cat[0] = lstm1.output.view({batch_, x.size(0), hidden_});
    cat[1] = torch::flip(lstm2.output.view({batch_, x.size(0), hidden_}), 1);
    // Feed into Linear Layer
    auto out = torch::sigmoid(
        linear->forward(cat.view({batch_, x.size(0), hidden_ * directions_})));
    //[batch,sequence,FEATURE]
    return out;
  }

private:
  long batch_{1};
  long directions_{1};
  long hidden_{1};
};

int main() {
  // Configurations
  static const int inputs = 1;
  static const int sequence = 3;
  static const int batch = 1;
  static const int layers = 3;
  static const int hidden = 2;
  static const int outputs = 1;
  static const int directions = 2;

  // Input: 0.1, 0.2, 0.3 -> Expected Output: 0.4, 0.5, 0.6
  BLSTM_Model model =
      BLSTM_Model(layers, hidden, inputs, outputs, batch, directions);
  torch::optim::Adam optimizer(model.parameters(),
                               torch::optim::AdamOptions(0.0001));
  // Input
  torch::Tensor input = torch::empty({sequence, inputs});
  auto input_acc = input.accessor<float, 2>();
  size_t count = 0;
  for (float i = 0.1; i < 0.4; i += 0.1) {
    input_acc[count][0] = i;
    count++;
  }
  // Target
  torch::Tensor target = torch::empty({sequence, outputs});
  auto target_acc = target.accessor<float, 2>();
  count = 0;
  for (float i = 0.4; i < 0.7; i += 0.1) {
    target_acc[count][0] = i;
    count++;
  }
  // Train
  for (size_t i = 0; i < 6000; i++) {
    torch::Tensor output = model.forward(input);
    auto loss = torch::mse_loss(output.view({sequence, outputs}), target);
    std::cout << "Loss " << i << " : " << loss.item<float>() << std::endl;
    loss.backward();
    optimizer.step();
  }
  // Test: Response should be about (0.4, 0.5, 0.6)
  torch::Tensor output = model.forward(input);
  std::cout << output << std::endl;
  return EXIT_SUCCESS;
}
