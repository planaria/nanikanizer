#pragma once

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244)
#endif

#include <cstdint>
#include <fstream>
#include <algorithm>
#include <memory>
#include <random>
#include <valarray>
#include <unordered_map>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <boost/dynamic_bitset.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/function_output_iterator.hpp>
#include <boost/optional.hpp>
#include <boost/utility/in_place_factory.hpp>
#include <boost/cast.hpp>

#include "expression.hpp"
#include "variable.hpp"
#include "evaluator.hpp"

#include "negate_expression.hpp"
#include "add_expression.hpp"
#include "subtract_expression.hpp"
#include "multiply_expression.hpp"
#include "divide_expression.hpp"
#include "abs_expression.hpp"
#include "square_expression.hpp"
#include "sqrt_expression.hpp"
#include "sum_expression.hpp"
#include "minmax_expression.hpp"
#include "norm_expression.hpp"
#include "sigmoid_expression.hpp"
#include "tanh_expression.hpp"
#include "relu_expression.hpp"
#include "softmax_expression.hpp"
#include "cross_entropy_expression.hpp"
#include "matrix_product_expression.hpp"
#include "matrix_transpose_expression.hpp"
#include "convolution_2d_expression.hpp"
#include "padding_2d_expression.hpp"
#include "spacing_2d_expression.hpp"
#include "skipping_2d_expression.hpp"
#include "max_pooling_2d_expression.hpp"
#include "sum_pooling_2d_expression.hpp"
#include "depth_concat_expression.hpp"
#include "depth_mean_expression.hpp"
#include "dropout_expression.hpp"

#include "linear_layer.hpp"
#include "bidirectional_linear_layer.hpp"
#include "dropout_layer.hpp"

#include "sgd_optimizer.hpp"
#include "adagrad_optimizer.hpp"
#include "rmsprop_optimizer.hpp"
#include "adadelta_optimizer.hpp"
#include "adam_optimizer.hpp"

#include "id_util.hpp"
#include "cifar_util.hpp"
