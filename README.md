# nanikanizer
An neural network library for C++

## Features
- Expression-based network description
- Header-only  
  `#include <nanikanizer/nanikanizer.hpp>`

### Expressions
- Four Arithmetic Operations
- Abs, Square, Square Root
- Sum, Min, Max, Norm, Squared Norm
- Sigmoid, Tanh, ReLU, Softmax
- Cross-Entropy Error
- Matrix Production, Matrix Transpose
- Convolution, Padding, Spacing, Skipping
- Max-Pooling, Sum-Pooling
- Depth Concat, Depth Mean

### Optimizers
- SGD
- AdaGrad
- RMSProp
- AdaDelta
- Adam

## Requirements

### Supported Compilers
- Clang (>= 3.7.0)
- Visual Studio 2015

### External Libraries
- Boost C++ Libraries (>= 1.60.0)
- [Catch](https://github.com/philsquared/Catch) (for test)

## License
The BSD 3-Clause License (see [LICENSE](LICENSE))
