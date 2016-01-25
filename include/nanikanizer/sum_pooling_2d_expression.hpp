#pragma once
#include "pooling_2d_expression.hpp"

namespace nnk
{

	struct sum_pooling_2d_operator
	{

		template <class T>
		static T forward(const T* x, std::size_t height, std::size_t width, std::size_t stride, std::size_t depth)
		{
			T result = T();

			std::size_t row_index = 0;

			for (std::size_t i = 0; i < height; ++i)
			{
				std::size_t index = row_index;

				for (std::size_t j = 0; j < width; ++j)
				{
					result += x[index];
					index += depth;
				}

				row_index += stride;
			}

			return result;
		}

		template <class T>
		static void backward(T* x_grad, const T* /*x*/, std::size_t height, std::size_t width, std::size_t stride, std::size_t depth, T /*y*/, T /*y_grad*/)
		{
			std::size_t row_index = 0;

			for (std::size_t i = 0; i < height; ++i)
			{
				std::size_t index = row_index;

				for (std::size_t j = 0; j < width; ++j)
				{
					x_grad[index] += static_cast<T>(1.0);
					index += depth;
				}

				row_index += stride;
			}
		}

	};

	template <class T>
	expression<T> sum_pooling_2d(
		const expression<T>& base,
		std::size_t input_height,
		std::size_t input_width,
		std::size_t input_depth,
		std::size_t filter_height,
		std::size_t filter_width)
	{
		return pooling_2d<sum_pooling_2d_operator>(
			base,
			input_height,
			input_width,
			input_depth,
			filter_height,
			filter_width);
	}

}
