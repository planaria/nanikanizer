#pragma once
#include "unary_operator_expression.hpp"
#include "math_util.hpp"

namespace nnk
{

	struct relu_operator
	{

		template <class T>
		static T forward(T x)
		{
			return relu(x);
		}

		template <class T>
		static void backward(T& x_grad, T x, T y, T y_grad)
		{
			if(x >= static_cast<T>(0.0))
				x_grad += y_grad;
		}

	};

	template <class T>
	expression<T> relu(const expression<T>& base)
	{
		return unary_operator<relu_operator>(base);
	}

}
