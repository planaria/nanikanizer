#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class padding_2d_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		padding_2d_expression_node(
			const node_pointer& base,
			std::size_t input_height,
			std::size_t input_width,
			std::size_t input_depth,
			std::size_t padding_top,
			std::size_t padding_left,
			std::size_t padding_bottom,
			std::size_t padding_right,
			scalar_type padding_value)
			: base_(base)
			, input_height_(input_height)
			, padding_top_(padding_top)
			, padding_left_(padding_left)
			, padding_bottom_(padding_bottom)
			, padding_right_(padding_right)
			, padding_value_(padding_value)
		{
			BOOST_ASSERT(input_height >= 1);
			BOOST_ASSERT(input_width >= 1);
			BOOST_ASSERT(input_depth >= 1);
			BOOST_ASSERT(padding_top >= 0);
			BOOST_ASSERT(padding_left >= 0);
			BOOST_ASSERT(padding_bottom >= 0);
			BOOST_ASSERT(padding_right >= 0);

			input_stride_ = input_width * input_depth;
			input_size_ = input_height * input_stride_;
			output_stride_ = (input_width + padding_left + padding_right) * input_depth;
			output_size_ = (input_height + padding_top + padding_bottom) * output_stride_;
			padding_initial_ = padding_top * output_stride_;
			padding_final_ = padding_bottom * output_stride_;
			padding_begin_ = padding_left * input_depth;
			padding_end_ = padding_right * input_depth;
		}

		padding_2d_expression_node(
			const node_pointer& base,
			std::size_t input_height,
			std::size_t input_width,
			std::size_t input_depth,
			std::size_t padding_height,
			std::size_t padding_width,
			scalar_type padding_value)
			: padding_2d_expression_node(
				base,
				input_height,
				input_width,
				input_depth,
				padding_height,
				padding_width,
				padding_height,
				padding_width,
				padding_value)
		{
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
			std::size_t input_index = 0;

			for (std::size_t i = 0; i < count; ++i)
			{
				for (std::size_t j = 0; j < padding_initial_; ++j)
					this->output()[output_index++] = padding_value_;

				for (std::size_t h = 0; h < input_height_; ++h)
				{
					for (std::size_t w = 0; w < padding_begin_; ++w)
						this->output()[output_index++] = padding_value_;

					for (std::size_t w = 0; w < input_stride_; ++w)
						this->output()[output_index++] = base_->output()[input_index++];

					for (std::size_t w = 0; w < padding_end_; ++w)
						this->output()[output_index++] = padding_value_;
				}

				for (std::size_t j = 0; j < padding_final_; ++j)
					this->output()[output_index++] = padding_value_;
			}

			BOOST_ASSERT(input_index == input_size_ * count);
		}

		virtual void backward() override
		{
			BOOST_ASSERT(base_->output().size() % input_size_ == 0);
			std::size_t count = base_->output().size() / input_size_;

			std::size_t output_index = 0;
			std::size_t input_index = 0;

			for (std::size_t i = 0; i < count; ++i)
			{
				output_index += padding_initial_;

				for (std::size_t h = 0; h < input_height_; ++h)
				{
					output_index += padding_begin_;

					for (std::size_t w = 0; w < input_stride_; ++w)
						base_->output_grad()[input_index++] += this->output_grad()[output_index++];

					output_index += padding_end_;
				}

				output_index += padding_final_;
			}

			BOOST_ASSERT(input_index == input_size_ * count);
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(base_.get());
		}

	private:

		node_pointer base_;

		std::size_t input_height_;
		std::size_t padding_top_;
		std::size_t padding_left_;
		std::size_t padding_bottom_;
		std::size_t padding_right_;
		scalar_type padding_value_;

		std::size_t input_stride_;
		std::size_t input_size_;

		std::size_t output_stride_;
		std::size_t output_size_;

		std::size_t padding_initial_;
		std::size_t padding_final_;
		std::size_t padding_begin_;
		std::size_t padding_end_;

	};

	template <class T>
	expression<T> padding_2d(
		const expression<T>& base,
		std::size_t input_height,
		std::size_t input_width,
		std::size_t input_depth,
		std::size_t padding_height,
		std::size_t padding_width,
		T padding_value = T())
	{
		return expression<T>(std::make_shared<padding_2d_expression_node<T>>(
			base.root(),
			input_height,
			input_width,
			input_depth,
			padding_height,
			padding_width,
			padding_value));
	}

	template <class T>
	expression<T> padding_2d(
		const expression<T>& base,
		std::size_t input_height,
		std::size_t input_width,
		std::size_t input_depth,
		std::size_t padding_top,
		std::size_t padding_left,
		std::size_t padding_bottom,
		std::size_t padding_right,
		T padding_value = T())
	{
		return expression<T>(std::make_shared<padding_2d_expression_node<T>>(
			base.root(),
			input_height,
			input_width,
			input_depth,
			padding_top,
			padding_left,
			padding_bottom,
			padding_right,
			padding_value));
	}

}
