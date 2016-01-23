#pragma once
#include "expression.hpp"
#include "sum_expression.hpp"
#include "square_expression.hpp"
#include "sqrt_expression.hpp"

namespace nnk
{

	template <class T>
	expression<T> norm_sq(const expression<T>& base)
	{
		return sum(square(base));
	}

	template <class T>
	expression<T> norm(const expression<T>& base)
	{
		return sqrt(norm_sq(base));
	}

}
