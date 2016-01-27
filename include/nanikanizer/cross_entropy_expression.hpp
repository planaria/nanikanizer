#pragma once
#include "unary_operator_expression.hpp"
#include "sum_expression.hpp"
#include "math_util.hpp"

namespace nnk
{

	struct cross_entropy_operator
	{

		template <class T>
		static T forward(T x)
		{
			x = clamp(x, static_cast<T>(-0.999), static_cast<T>(0.999));

			if (x < static_cast<T>(0.0))
				return -std::log1p(x);
			else
				return -std::log1p(-x);
		}

		template <class T>
		static void backward(T& x_grad, T x, T /*y*/, T y_grad)
		{
			x = clamp(x, static_cast<T>(-0.999), static_cast<T>(0.999));

			if (x < static_cast<T>(0.0))
				x_grad -= y_grad / (static_cast<T>(1.0) + x);
			else
				x_grad += y_grad / (static_cast<T>(1.0) - x);
		}

	};

	template <class T>
	expression<T> cross_entropy(const expression<T>& base)
	{
		return sum(unary_operator<cross_entropy_operator>(base));
	}

}
