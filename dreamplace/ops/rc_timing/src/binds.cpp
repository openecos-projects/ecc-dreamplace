#include <torch/extension.h> // Includes pybind11

// Include headers for all operators
#include "rc_timing.hh"

// Define the module entry point
// TORCH_EXTENSION_NAME is defined during the build process (in setup.py)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Add docstring for the module (optional)
  m.doc() = "Python bindings for custom RC Timing C++ operators";

  // Register functions for Load Operator
  m.def("load_forward_cpp",           // Python name
        &load_forward_cpp,            // C++ function pointer
        "Elmore Load Forward (C++)"); // Docstring

  m.def("load_backward_cpp", &load_backward_cpp, "Elmore Load Backward (C++)");

  // Register functions for Delay Operator
  // Note: Ensure the function pointers match the declared functions
  m.def("delay_forward_cpp", &delay_forward_cpp, "Elmore Delay Forward (C++)");

  // Important: Backward functions returning std::vector<at::Tensor> are bound
  // directly
  m.def("delay_backward_cpp", &delay_backward_cpp,
        "Elmore Delay Backward (C++)");

  // Register functions for LDelay Operator
  m.def("ldelay_forward_cpp", &ldelay_forward_cpp, "LDelay Forward (C++)");

  m.def("ldelay_backward_cpp", &ldelay_backward_cpp, "LDelay Backward (C++)");

  // Register functions for Beta Operator
  m.def("beta_forward_cpp", &beta_forward_cpp, "Beta Forward (C++)");

  m.def("beta_backward_cpp", &beta_backward_cpp, "Beta Backward (C++)");

  // Add more function registrations if needed
}