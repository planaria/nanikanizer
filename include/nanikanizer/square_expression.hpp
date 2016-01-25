#pragma once
#include "unary_operator_expression.hpp"
#include "math_util.hpp"

namespace nnk
{

	struct square_operator
	{

		template <class T>
		static T forward(T x)
		{
			return square(x);
		}

		template <class T>
		static void backward(T& x_grad, T x, T /*y*/, T y_grad)
		{
			x_grad += static_cast<T>(2.0) * x * y_grad;
		}

	};

	template <class T>
	expression<T> square(const expression<T>& base)
	{
		return unary_operator<square_operator>(base);
	}

}
