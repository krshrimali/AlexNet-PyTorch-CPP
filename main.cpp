/* Author: Kushashwa, http://www.github.com/krshrimali */

#include <torch/torch.h>

/* Sample code for implementation of AlexNet using PyTorch C++ API */

struct Net: torch::nn::Module {
    // AlexNet Layer
    Net() {
        // Initialize AlexNet
        conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, 11).padding(1)));
        // Maxpool2d Layer
        conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, 5).padding(2)));
        // Overlapping Maxpool2d Layer
        conv1_3 = register_module("conv1_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, 3).padding(1)));
        conv1_4 = register_module("conv1_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, 3).padding(1)));
        conv1_5 = register_module("conv1_5", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, 3).padding(1)));
        // Overlapping Maxpool2d Layer
        // FC1 - FC2- Softmax
        fc1 = register_module("fc1", torch::nn::Linear(9216, 4096));
        fc2 = register_module("fc2", torch::nn::Linear(4096, 4096));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::max_pool2d(conv1_1->forward(x));
        x = torch::max_pool2d(conv1_2->forward(x));
        x = torch::max_pool2d(conv1_5->forward(x));
        x = fc1->forward(x);
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::Conv2d conv1_3{nullptr};
    torch::nn::Conv2d conv1_4{nullptr};
    torch::nn::Conv2d conv1_5{nullptr};

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
    auto net = std::make_shared<Net>();

    // Create multi-threaded data loader for MNIST data
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(torch::data::datasets::MNIST("../../data").map(torch::data::transforms::Normalize<>(0.13707, 0.3081)).map(
                    torch::data::transforms::Stack<>())), 64);
    torch::optim::SGD optimizer(net->parameters(), 0.01); // Learning Rate 0.01

    // net.train();

    for(size_t epoch=1; epoch<=10; ++epoch) {
        size_t batch_index = 0;
        // Iterate data loader to yield batches from the dataset
        for (auto& batch: *data_loader) {
            // Reset gradients
            optimizer.zero_grad();
            // Execute the model
            torch::Tensor prediction = net->forward(batch.data);
            // Compute loss value
            torch::Tensor loss = torch::nll_loss(prediction, batch.target);
            // Compute gradients
            loss.backward();
            // Update the parameters
            optimizer.step();

            // Output the loss and checkpoint every 100 batches
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                torch::save(net, "net.pt");
            }
        }
    }
}

