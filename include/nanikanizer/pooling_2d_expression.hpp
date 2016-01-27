#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T, class OP>
	class pooling_2d_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		pooling_2d_expression_node(
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
			BOOST_ASSERT(input_height >= 2);
			BOOST_ASSERT(input_width >= 2);
			BOOST_ASSERT(input_depth >= 1);
			BOOST_ASSERT(filter_height >= 2);
			BOOST_ASSERT(filter_width >= 2);
			BOOST_ASSERT(input_height % filter_height == 0);
			BOOST_ASSERT(input_width % filter_width == 0);

			input_stride_ = input_width * input_depth;
			input_row_span_ = filter_height_ * input_stride_;
			input_col_span_ = filter_width * input_depth;
			input_size_ = input_height * input_stride_;
			output_height_ = input_height / filter_height;
			output_width_ = input_width / filter_width;
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

			std::size_t total_output_size = count * output_size_;

			if (this->output().size() != total_output_size)
				this->output().resize(total_output_size);

			std::size_t output_index = 0;
			std::size_t base_row_index = 0;

			for (std::size_t i = 0; i < count; ++i)
			{
				for (std::size_t h = 0; h < output_height_; ++h)
				{
					std::size_t base_index = base_row_index;

					for (std::size_t w = 0; w < output_width_; ++w)
					{
						for (std::size_t d = 0; d < input_depth_; ++d)
						{
							this->output()[output_index] = OP::forward(
								&base_->output()[base_index + d],
								filter_height_,
								filter_width_,
								input_stride_,
								input_depth_);

							++output_index;
						}

						base_index += input_col_span_;
					}

					base_row_index += input_row_span_;
				}
			}
		}

		virtual void backward() override
		{
			BOOST_ASSERT(base_->output().size() % input_size_ == 0);
			std::size_t count = base_->output().size() / input_size_;

			std::size_t output_index = 0;
			std::size_t base_row_index = 0;

			for (std::size_t i = 0; i < count; ++i)
			{
				for (std::size_t h = 0; h < output_height_; ++h)
				{
					std::size_t base_index = base_row_index;

					for (std::size_t w = 0; w < output_width_; ++w)
					{
						for (std::size_t d = 0; d < input_depth_; ++d)
						{
							OP::backward(
								&base_->output_grad()[base_index + d],
								&base_->output()[base_index + d],
								filter_height_,
								filter_width_,
								input_stride_,
								input_depth_,
								this->output()[output_index],
								this->output_grad()[output_index]);

							++output_index;
						}

						base_index += input_col_span_;
					}

					base_row_index += input_row_span_;
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
		std::size_t input_row_span_;
		std::size_t input_col_span_;
		std::size_t input_size_;

		std::size_t output_height_;
		std::size_t output_width_;
		std::size_t output_size_;

	};

	template <class OP, class T>
	expression<T> pooling_2d(
		const expression<T>& base,
		std::size_t input_height,
		std::size_t input_width,
		std::size_t input_depth,
		std::size_t filter_height,
		std::size_t filter_width)
	{
		return expression<T>(std::make_shared<pooling_2d_expression_node<T, OP>>(
			base.root(),
			input_height,
			input_width,
			input_depth,
			filter_height,
			filter_width));
	}

}
