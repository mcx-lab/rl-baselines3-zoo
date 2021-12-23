#include <torch/script.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "please specify path to traced model" << std::endl;
    return -1;
  }

  std::cout << __cplusplus << std::endl;
  torch::jit::script::Module feature_extractor;
  torch::jit::script::Module mlp_extractor;
  torch::jit::script::Module action_net;
  torch::jit::script::Module value_net;
  try {
    std::string model_dir = argv[1];
    std::string fe_file = "feature_extractor.pt";
    std::string mlp_file = "mlp_extractor.pt";
    std::string an_file = "action_net.pt";
    std::string vn_file = "value_net.pt";

    // deserialise the ScriptModules
    feature_extractor = torch::jit::load(model_dir + fe_file);
    mlp_extractor = torch::jit::load(model_dir + mlp_file);
    action_net = torch::jit::load(model_dir + an_file);
    value_net = torch::jit::load(model_dir + vn_file);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model" << std::endl;
    return -1;
  }

  std::cout << "done loading the model" << std::endl;
}

