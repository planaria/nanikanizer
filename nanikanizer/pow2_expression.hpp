#pragma once
#include "unary_operator_expression.hpp"
#include "math_util.hpp"

namespace nnk
{

	struct pow2_operator
	{

		template <class T>
		static T forward(T x)
		{
			return pow2(x);
		}

		template <class T>
		static void backward(T& x_grad, T x, T /*y*/, T y_grad)
		{
			x_grad += static_cast<T>(2.0) * x * y_grad;
		}

	};

	template <class T>
	expression<T> pow2(const expression<T>& base)
	{
		return unary_operator<pow2_operator>(base);
	}

}
