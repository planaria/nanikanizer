#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class matrix_transpose_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		matrix_transpose_expression_node(const node_pointer& base, std::size_t rows, std::size_t cols)
			: base_(base)
			, rows_(rows)
			, cols_(cols)
			, size_(rows * cols)
		{
			BOOST_ASSERT(rows >= 1);
			BOOST_ASSERT(cols >= 1);
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			if (this->output().size() != base_->output().size())
				this->output().resize(base_->output().size());

			BOOST_ASSERT(base_->output().size() % size_ == 0);
			std::size_t count = base_->output().size() / size_;

			std::size_t base_index = 0;

			for (std::size_t i = 0; i < count; ++i)
			{
				std::size_t output_col_index = base_index;

				for (std::size_t j = 0; j < rows_; ++j)
				{
					std::size_t output_index = output_col_index;

					for (std::size_t k = 0; k < cols_; ++k)
					{
						this->output()[output_index] = base_->output()[base_index];

						++base_index;
						output_index += rows_;
					}

					++output_col_index;
				}
			}
		}

		virtual void backward() override
		{
			BOOST_ASSERT(base_->output().size() % size_ == 0);
			std::size_t count = base_->output().size() / size_;

			std::size_t base_index = 0;

			for (std::size_t i = 0; i < count; ++i)
			{
				std::size_t output_col_index = base_index;

				for (std::size_t j = 0; j < rows_; ++j)
				{
					std::size_t output_index = output_col_index;

					for (std::size_t k = 0; k < cols_; ++k)
					{
						base_->output_grad()[base_index] += this->output_grad()[output_index];

						++base_index;
						output_index += rows_;
					}

					++output_col_index;
				}
			}
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(base_.get());
		}

	private:

		node_pointer base_;

		std::size_t rows_;
		std::size_t cols_;
		std::size_t size_;

	};

	template <class T>
	expression<T> matrix_transpose(const expression<T>& base, std::size_t rows, std::size_t cols)
	{
		return expression<T>(std::make_shared<matrix_transpose_expression_node<T>>(base.root(), rows, cols));
	}

}
