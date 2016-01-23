#pragma once

#include <cstdint>
#include <algorithm>
#include <memory>
#include <random>
#include <valarray>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/function_output_iterator.hpp>
#include <boost/operators.hpp>
#include <boost/optional.hpp>
#include <boost/utility/in_place_factory.hpp>

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
#include "norm_expression.hpp"
#include "sigmoid_expression.hpp"
#include "sigmoid_cross_entropy_expression.hpp"
#include "relu_expression.hpp"
#include "tanh_expression.hpp"
#include "matrix_product_expression.hpp"

#include "linear_layer.hpp"

#include "sgd_optimizer.hpp"
#include "adagrad_optimizer.hpp"
#include "rmsprop_optimizer.hpp"
#include "adadelta_optimizer.hpp"
#include "adam_optimizer.hpp"
