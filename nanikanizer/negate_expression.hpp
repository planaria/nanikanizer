#pragma once
#include "unary_operator_expression.hpp"

namespace nnk
{

	struct negate_operator
	{

		template <class T>
		static T forward(T x)
		{
			return -x;
		}

		template <class T>
		static void backward(T& x_grad, T /*x*/, T /*y*/, T y_grad)
		{
			x_grad -= y_grad;
		}

	};

	template <class T>
	expression<T> operator -(const expression<T>& base)
	{
		return unary_operator<negate_operator>(base);
	}

}
