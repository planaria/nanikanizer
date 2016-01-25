#pragma once
#include "unary_operator_expression.hpp"

namespace nnk
{

	struct abs_operator
	{

		template <class T>
		static T forward(T x)
		{
			return std::abs(x);
		}

		template <class T>
		static void backward(T& x_grad, T x, T /*y*/, T y_grad)
		{
			if(x < static_cast<T>(0.0))
				x_grad -= y_grad;
			else
				x_grad += y_grad;
		}

	};

	template <class T>
	expression<T> abs(const expression<T>& base)
	{
		return unary_operator<abs_operator>(base);
	}

}
