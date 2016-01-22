#pragma once
#include "expression.hpp"
#include "sum_expression.hpp"
#include "pow2_expression.hpp"
#include "sqrt_expression.hpp"

namespace nnk
{

	template <class T>
	expression<T> norm_sq(const expression<T>& base)
	{
		return sum(pow2(base));
	}

	template <class T>
	expression<T> norm(const expression<T>& base)
	{
		return sqrt(norm_sq(base));
	}

}
