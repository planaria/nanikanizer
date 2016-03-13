#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class convolution_2d_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		convolution_2d_expression_node(
			const node_pointer& base,
			std::size_t input_height,
			std::size_t input_width,
			std::size_t input_depth,
			std::size_t filter_height,
			std::size_t filter_width)
			: base_(base)
			, input_height_(input_height)
			, input_width_(input_width)
			, input_depth_(input_depth)
			, filter_height_(filter_height)
			, filter_width_(filter_width)
		{
			BOOST_ASSERT(input_height >= 1);
			BOOST_ASSERT(input_width >= 1);
			BOOST_ASSERT(input_depth >= 1);
			BOOST_ASSERT(filter_height >= 1);
			BOOST_ASSERT(filter_width >= 1);
			BOOST_ASSERT(input_height >= filter_height);
			BOOST_ASSERT(input_width >= filter_width);

			input_stride_ = input_width * input_depth;
			input_size_ = input_height * input_stride_;
			output_height_ = input_height - filter_height + 1;
			output_width_ = input_width - filter_width + 1;
			filter_stride_ = filter_width * input_depth;
			output_depth_ = filter_height * filter_stride_;
			output_size_ = output_height_ * output_width_ * output_depth_;
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			BOOST_ASSERT(base_->output().size() % input_size_ == 0);
			std::size_t count = base_->output().size() / input_size_;

			std::size_t total_output_size = output_size_ * count;

			if (this->output().size() != total_output_size)
				this->output().resize(total_output_size);

			#pragma omp parallel for schedule(static)
			for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i)
			{
				std::size_t output_index = i * output_size_;
				std::size_t base_row_index = i * input_size_;

				for (std::size_t h = 0; h < output_height_; ++h)
				{
					std::size_t base_element_index = base_row_index;

					for (std::size_t w = 0; w < output_width_; ++w)
					{
						std::size_t filter_row_index = base_element_index;

						for (std::size_t k = 0; k < filter_height_; ++k)
						{
							for (std::size_t l = 0; l < filter_stride_; ++l)
							{
								this->output()[output_index] = base_->output()[filter_row_index + l];
								++output_index;
							}

							filter_row_index += input_stride_;
						}

						base_element_index += input_depth_;
					}

					base_row_index += input_stride_;
				}
			}
		}

		virtual void backward() override
		{
			BOOST_ASSERT(base_->output().size() % input_size_ == 0);
			std::size_t count = base_->output().size() / input_size_;

			#pragma omp parallel for schedule(static)
			for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i)
			{
				std::size_t output_index = i * output_size_;
				std::size_t base_row_index = i * input_size_;

				for (std::size_t h = 0; h < output_height_; ++h)
				{
					std::size_t base_element_index = base_row_index;

					for (std::size_t w = 0; w < output_width_; ++w)
					{
						std::size_t filter_row_index = base_element_index;

						for (std::size_t k = 0; k < filter_height_; ++k)
						{
							for (std::size_t l = 0; l < filter_stride_; ++l)
							{
								base_->output_grad()[filter_row_index + l] += this->output_grad()[output_index];
								++output_index;
							}

							filter_row_index += input_stride_;
						}

						base_element_index += input_depth_;
					}

					base_row_index += input_stride_;
				}
			}
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(base_.get());
		}

	private:

		node_pointer base_;

		std::size_t input_height_;
		std::size_t input_width_;
		std::size_t input_depth_;
		std::size_t filter_height_;
		std::size_t filter_width_;

		std::size_t input_stride_;
		std::size_t input_size_;
		std::size_t output_height_;
		std::size_t output_width_;
		std::size_t output_depth_;
		std::size_t output_size_;
		std::size_t filter_stride_;

	};

	template <class T>
	expression<T> convolution_2d(
		const expression<T>& base,
		std::size_t input_height,
		std::size_t input_width,
		std::size_t input_depth,
		std::size_t filter_height,
		std::size_t filter_width)
	{
		return expression<T>(std::make_shared<convolution_2d_expression_node<T>>(
			base.root(),
			input_height,
			input_width,
			input_depth,
			filter_height,
			filter_width));
	}

}
