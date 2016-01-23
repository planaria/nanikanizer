#pragma once
#include "unary_operator_expression.hpp"
#include "math_util.hpp"

namespace nnk
{

	struct sigmoid_cross_entropy_operator
	{

		template <class T>
		static T forward(T x)
		{
			if (x < static_cast<T>(0.0))
				return -std::log1p(std::exp(x));
			else
				return -std::log1p(std::exp(-x));
		}

		template <class T>
		static void backward(T& x_grad, T x, T /*y*/, T y_grad)
		{
			if (x < static_cast<T>(0.0))
				x_grad += y_grad * (sigmoid(-x) - static_cast<T>(1.0));
			else
				x_grad += y_grad * sigmoid(-x);
		}

	};

	template <class T>
	expression<T> sigmoid_cross_entropy(const expression<T>& base)
	{
		return unary_operator<sigmoid_cross_entropy_operator>(base);
	}

}
