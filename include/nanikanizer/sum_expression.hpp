#pragma once
#include "block_operator_expression.hpp"

namespace nnk
{

	struct sum_operator
	{

		template <class T>
		static T forward(const T* x, std::size_t block_size)
		{
			T result = T();

			for (std::size_t i = 0; i < block_size; ++i)
				result += x[i];

			return result;
		}

		template <class T>
		static void backward(T* x_grad, const T* /*x*/, std::size_t block_size, T /*y*/, T y_grad)
		{
			for (std::size_t i = 0; i < block_size; ++i)
				x_grad[i] += y_grad;
		}

	};

	template <class T>
	expression<T> sum(const expression<T>& base, const boost::optional<std::size_t>& block_size = boost::none)
	{
		return block_operator<sum_operator>(base, block_size);
	}

}
