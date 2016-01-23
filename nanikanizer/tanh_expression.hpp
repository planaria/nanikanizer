#pragma once
#include "unary_operator_expression.hpp"
#include "math_util.hpp"

namespace nnk
{

	struct tanh_operator
	{

		template <class T>
		static T forward(T x)
		{
			return std::tanh(x);
		}

		template <class T>
		static void backward(T& x_grad, T x, T y, T y_grad)
		{
			x_grad += (static_cast<T>(1.0) - y * y) * y_grad;
		}

	};

	template <class T>
	expression<T> tanh(const expression<T>& base)
	{
		return unary_operator<tanh_operator>(base);
	}

}
