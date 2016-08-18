#pragma once
#include "binary_operator_expression.hpp"

namespace nnk
{

	struct subtract_operator
	{

		template <class T>
		static T forward(T lhs, T rhs)
		{
			return lhs - rhs;
		}

		template <class T>
		static void backward(T& lhs_grad, T& rhs_grad, T /*lhs*/, T /*rhs*/, T /*y*/, T y_grad)
		{
			lhs_grad += y_grad;
			rhs_grad -= y_grad;
		}

	};

	template <class T>
	expression<T> operator -(const expression<T>& lhs, const expression<T>& rhs)
	{
		return binary_operator<subtract_operator>(lhs, rhs);
	}

	template <class T>
	expression<T>& operator -=(expression<T>& lhs, const expression<T>& rhs)
	{
		lhs = lhs - rhs;
		return lhs;
	}

	template <class T>
	expression<T> operator -(const expression<T>& lhs, T rhs)
	{
		return lhs - expression<T>({ rhs });
	}

	template <class T>
	expression<T> operator -(T lhs, const expression<T>& rhs)
	{
		return expression<T>({ lhs }) - rhs;
	}

	template <class T>
	expression<T>& operator -=(expression<T>& lhs, T rhs)
	{
		lhs = lhs - rhs;
		return lhs;
	}

}
