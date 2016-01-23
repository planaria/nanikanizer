#pragma once
#include "expression.hpp"
#include "sum_expression.hpp"
#include "square_expression.hpp"
#include "sqrt_expression.hpp"

namespace nnk
{

	template <class T>
	expression<T> norm_sq(const expression<T>& base, const boost::optional<std::size_t>& block_size = boost::none)
	{
		return sum(square(base), block_size);
	}

	template <class T>
	expression<T> norm(const expression<T>& base, const boost::optional<std::size_t>& block_size = boost::none)
	{
		return sqrt(norm_sq(base, block_size));
	}

}
