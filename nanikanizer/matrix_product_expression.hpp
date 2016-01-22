#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class matrix_product_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base;

	public:

		typedef typename base::scalar_type scalar_type;
		typedef typename base::tensor_type tensor_type;
		typedef typename base::node_pointer node_pointer;

		matrix_product_expression_node(
			const node_pointer& lhs,
			const node_pointer& rhs,
			std::size_t lhs_rows,
			std::size_t lhs_cols,
			std::size_t rhs_rows,
			std::size_t rhs_cols)
			: lhs_(lhs)
			, rhs_(rhs)
			, lhs_rows_(lhs_rows)
			, lhs_cols_(lhs_cols)
			, rhs_cols_(rhs_cols)
			, lhs_size_(lhs_rows * lhs_cols)
			, rhs_size_(rhs_rows * rhs_cols)
			, output_size_(lhs_rows * rhs_cols)
		{
			BOOST_ASSERT(lhs_cols == rhs_rows);
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			BOOST_ASSERT(lhs_->output().size() == lhs_size_);
			BOOST_ASSERT(rhs_->output().size() == rhs_size_);

			if (this->output().size() != output_size_)
				this->output().resize(output_size_);

			std::size_t lhs_row_index = 0;
			std::size_t output_index = 0;

			for (std::size_t i = 0; i < lhs_rows_; ++i)
			{
				std::size_t rhs_col_index = 0;

				for (std::size_t j = 0; j < rhs_cols_; ++j)
				{
					std::size_t lhs_index = lhs_row_index;
					std::size_t rhs_index = rhs_col_index;

					this->output()[output_index] = static_cast<scalar_type>(0.0);

					for (std::size_t k = 0; k < lhs_cols_; ++k)
					{
						this->output()[output_index] += lhs_->output()[lhs_index] * rhs_->output()[rhs_index];

						++lhs_index;
						rhs_index += rhs_cols_;
					}

					++rhs_col_index;
					++output_index;
				}

				lhs_row_index += lhs_cols_;
			}
		}

		virtual void backward() override
		{
			std::size_t lhs_row_index = 0;
			std::size_t output_index = 0;

			for (std::size_t i = 0; i < lhs_rows_; ++i)
			{
				std::size_t rhs_col_index = 0;

				for (std::size_t j = 0; j < rhs_cols_; ++j)
				{
					std::size_t lhs_index = lhs_row_index;
					std::size_t rhs_index = rhs_col_index;

					for (std::size_t k = 0; k < lhs_cols_; ++k)
					{
						lhs_->output_grad()[lhs_index] += this->output_grad()[output_index] * rhs_->output()[rhs_index];
						rhs_->output_grad()[rhs_index] += this->output_grad()[output_index] * lhs_->output()[lhs_index];

						++lhs_index;
						rhs_index += rhs_cols_;
					}

					++rhs_col_index;
					++output_index;
				}

				lhs_row_index += lhs_cols_;
			}
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(lhs_.get());
			callback(rhs_.get());
		}

	private:

		node_pointer lhs_;
		node_pointer rhs_;

		std::size_t lhs_rows_;
		std::size_t lhs_cols_;
		std::size_t rhs_cols_;

		std::size_t lhs_size_;
		std::size_t rhs_size_;
		std::size_t output_size_;

	};

	template <class T>
	expression<T> matrix_product(
		const expression<T>& lhs,
		const expression<T>& rhs,
		std::size_t lhs_rows,
		std::size_t lhs_cols,
		std::size_t rhs_rows,
		std::size_t rhs_cols)
	{
		return expression<T>(std::make_shared<matrix_product_expression_node<T>>(
			lhs.root(), rhs.root(),
			lhs_rows, lhs_cols,
			rhs_rows, rhs_cols));
	}

}
