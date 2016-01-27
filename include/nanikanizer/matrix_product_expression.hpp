#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class matrix_product_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

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
			if (lhs_->output().size() == lhs_size_ &&
				rhs_->output().size() == rhs_size_)
			{
				if (this->output().size() != output_size_)
					this->output().resize(output_size_);

				forward_impl(
					&lhs_->output()[0],
					&rhs_->output()[0],
					&this->output()[0]);
			}
			else if (lhs_->output().size() == lhs_size_)
			{
				BOOST_ASSERT(rhs_->output().size() % rhs_size_ == 0);

				std::size_t count = rhs_->output().size() / rhs_size_;
				std::size_t total_output_size = output_size_ * count;

				if (this->output().size() != total_output_size)
					this->output().resize(total_output_size);

				#pragma omp parallel for schedule(static)
				for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i)
				{
					forward_impl(
						&lhs_->output()[0],
						&rhs_->output()[rhs_size_ * i],
						&this->output()[output_size_ * i]);
				}
			}
			else if (rhs_->output().size() == rhs_size_)
			{
				BOOST_ASSERT(lhs_->output().size() % lhs_size_ == 0);

				std::size_t count = lhs_->output().size() / lhs_size_;
				std::size_t total_output_size = output_size_ * count;

				if (this->output().size() != total_output_size)
					this->output().resize(total_output_size);

				#pragma omp parallel for schedule(static)
				for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i)
				{
					forward_impl(
						&lhs_->output()[lhs_size_ * i],
						&rhs_->output()[0],
						&this->output()[output_size_ * i]);
				}
			}
			else
			{
				BOOST_ASSERT(false);
			}
		}

		virtual void backward() override
		{
			if (lhs_->output().size() == lhs_size_ &&
				rhs_->output().size() == rhs_size_)
			{
				backward_impl(
					&lhs_->output_grad()[0],
					&rhs_->output_grad()[0],
					&lhs_->output()[0],
					&rhs_->output()[0],
					&this->output_grad()[0]);
			}
			else if (lhs_->output().size() == lhs_size_)
			{
				std::size_t count = rhs_->output().size() / rhs_size_;

				#pragma omp parallel for schedule(static)
				for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i)
				{
					backward_impl(
						&lhs_->output_grad()[0],
						&rhs_->output_grad()[rhs_size_ * i],
						&lhs_->output()[0],
						&rhs_->output()[rhs_size_ * i],
						&this->output_grad()[output_size_ * i]);
				}
			}
			else if (rhs_->output().size() == rhs_size_)
			{
				std::size_t count = lhs_->output().size() / lhs_size_;

				#pragma omp parallel for schedule(static)
				for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i)
				{
					backward_impl(
						&lhs_->output_grad()[lhs_size_ * i],
						&rhs_->output_grad()[0],
						&lhs_->output()[lhs_size_ * i],
						&rhs_->output()[0],
						&this->output_grad()[output_size_ * i]);
				}
			}
			else
			{
				BOOST_ASSERT(false);
			}
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(lhs_.get());
			callback(rhs_.get());
		}

	private:

		void forward_impl(const scalar_type* lhs, const scalar_type* rhs, scalar_type* output) const
		{
			if (rhs_cols_ == 1)
			{
				std::size_t lhs_row_index = 0;

				for (std::size_t i = 0; i < lhs_rows_; ++i)
				{
					output[i] = static_cast<scalar_type>(0.0);

					for (std::size_t k = 0; k < lhs_cols_; ++k)
						output[i] += lhs[lhs_row_index + k] * rhs[k];

					lhs_row_index += lhs_cols_;
				}
			}
			else
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

						output[output_index] = static_cast<scalar_type>(0.0);

						for (std::size_t k = 0; k < lhs_cols_; ++k)
						{
							output[output_index] += lhs[lhs_index] * rhs[rhs_index];

							++lhs_index;
							rhs_index += rhs_cols_;
						}

						++rhs_col_index;
						++output_index;
					}

					lhs_row_index += lhs_cols_;
				}
			}
		}

		void backward_impl(scalar_type* lhs_grad, scalar_type* rhs_grad, const scalar_type* lhs, const scalar_type* rhs, const scalar_type* grad) const
		{
			if (rhs_cols_ == 1)
			{
				std::size_t lhs_row_index = 0;

				for (std::size_t i = 0; i < lhs_rows_; ++i)
				{
					std::size_t lhs_index = lhs_row_index;
					scalar_type g = grad[i];

					for (std::size_t k = 0; k < lhs_cols_; ++k)
					{
						lhs_grad[lhs_row_index + k] += g * rhs[k];
						rhs_grad[k] += g * lhs[lhs_row_index + k];
					}

					lhs_row_index += lhs_cols_;
				}
			}
			else
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
							lhs_grad[lhs_index] += grad[output_index] * rhs[rhs_index];
							rhs_grad[rhs_index] += grad[output_index] * lhs[lhs_index];

							++lhs_index;
							rhs_index += rhs_cols_;
						}

						++rhs_col_index;
						++output_index;
					}

					lhs_row_index += lhs_cols_;
				}
			}
		}

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
