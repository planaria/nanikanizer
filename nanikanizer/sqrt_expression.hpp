#pragma once
#include "unary_operator_expression.hpp"

namespace nnk
{

	struct sqrt_operator
	{

		template <class T>
		static T forward(T x)
		{
			return std::sqrt(x);
		}

		template <class T>
		static void backward(T& x_grad, T x, T y, T y_grad)
		{
			x_grad += static_cast<T>(0.5) * y_grad / std::sqrt(x);
		}

	};

	template <class T>
	expression<T> sqrt(const expression<T>& base)
	{
		return unary_operator<sqrt_operator>(base);
	}

}
