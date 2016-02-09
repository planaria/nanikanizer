#pragma once
#include "block_operator_expression.hpp"

namespace nnk
{

	struct min_operator
	{

		template <class T>
		static T forward(const T* x, std::size_t block_size)
		{
			T result = x[0];

			for (std::size_t i = 1; i < block_size; ++i)
				result = std::min(result, x[i]);

			return result;
		}

		template <class T>
		static void backward(T* x_grad, const T* x, std::size_t block_size, T y, T y_grad)
		{
			for (std::size_t i = 0; i < block_size; ++i)
			{
				if (x[i] == y)
					x_grad[i] += y_grad;
			}
		}

	};

	template <class T>
	expression<T> min(const expression<T>& base, const boost::optional<std::size_t>& block_size = boost::none)
	{
		return block_operator<min_operator>(base, block_size);
	}

	struct max_operator
	{

		template <class T>
		static T forward(const T* x, std::size_t block_size)
		{
			T result = x[0];

			for (std::size_t i = 1; i < block_size; ++i)
				result = std::max(result, x[i]);

			return result;
		}

		template <class T>
		static void backward(T* x_grad, const T* x, std::size_t block_size, T y, T y_grad)
		{
			for (std::size_t i = 0; i < block_size; ++i)
			{
				if (x[i] == y)
					x_grad[i] += y_grad;
			}
		}

	};

	template <class T>
	expression<T> max(const expression<T>& base, const boost::optional<std::size_t>& block_size = boost::none)
	{
		return block_operator<max_operator>(base, block_size);
	}

}
