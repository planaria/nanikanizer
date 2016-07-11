#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class skipping_2d_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		skipping_2d_expression_node(
			const node_pointer& base,
			std::size_t input_height,
			std::size_t input_width,
			std::size_t input_depth,
			std::size_t skipping_height,
			std::size_t skipping_width)
			: base_(base)
			, input_height_(input_height)
			, input_width_(input_width)
			, input_depth_(input_depth)
			, skipping_height_(skipping_height)
			, skipping_width_(skipping_width)
		{
			BOOST_ASSERT(input_height >= 1);
			BOOST_ASSERT(input_width >= 1);
			BOOST_ASSERT(input_depth >= 1);

			input_size_ = input_width * input_height * input_depth;
			input_col_stride_ = (skipping_width + 1) * input_depth;
			input_row_stride_ = input_width * (skipping_height + 1) * input_depth;
			output_height_ = (input_height + 1) / (skipping_height + 1);
			output_width_ = (input_width + 1) / (skipping_width + 1);
			output_size_ = output_height_ * output_width_ * input_depth;
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
					std::size_t base_index = base_row_index;

					for (std::size_t w = 0; w < output_width_; ++w)
					{
						for (std::size_t d = 0; d < input_depth_; ++d)
							this->output()[output_index++] = base_->output()[base_index + d];

						base_index += input_col_stride_;
					}

					base_row_index += input_row_stride_;
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
					std::size_t base_index = base_row_index;

					for (std::size_t w = 0; w < output_width_; ++w)
					{
						for (std::size_t d = 0; d < input_depth_; ++d)
							base_->output_grad()[base_index + d] += this->output_grad()[output_index++];

						base_index += input_col_stride_;
					}

					base_row_index += input_row_stride_;
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
		std::size_t skipping_height_;
		std::size_t skipping_width_;

		std::size_t input_size_;
		std::size_t input_col_stride_;
		std::size_t input_row_stride_;
		std::size_t output_height_;
		std::size_t output_width_;
		std::size_t output_size_;

	};

	template <class T>
	expression<T> skipping_2d(
		const expression<T>& base,
		std::size_t input_height,
		std::size_t input_width,
		std::size_t input_depth,
		std::size_t skipping_height,
		std::size_t skipping_width)
	{
		return expression<T>(std::make_shared<skipping_2d_expression_node<T>>(
			base.root(),
			input_height,
			input_width,
			input_depth,
			skipping_height,
			skipping_width));
	}

}
