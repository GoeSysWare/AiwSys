
#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
	torch::DeviceType device_type = torch::kCUDA;  //torch::kCUDA  and torch::kCPU
	torch::Device device(device_type, 0);

	torch::Tensor tensor_rotation = torch::tensor({ 0.999832,0.0,-0.015292,0.00086496258196527018,0.99999655211205762,-0.0026255536555242422,
													0.015294316876925231,0.0026259908600024411,0.99982955268435725 }, at::kDouble).view({ 3,3 }).to(device);   //rotation
  return 0;
}
