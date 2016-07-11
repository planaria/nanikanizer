#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class depth_mean_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		depth_mean_expression_node(
			const node_pointer& base,
			std::size_t block_size)
			: base_(base)
			, block_size_(block_size)
		{
			BOOST_ASSERT(block_size >= 1);
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			BOOST_ASSERT(base_->output().size() % block_size_ == 0);
			std::size_t count = base_->output().size() / block_size_;

			if (this->output().size() != block_size_)
				this->output().resize(block_size_);

			this->output() = static_cast<scalar_type>(0.0);

			#pragma omp parallel for schedule(static)
			for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(block_size_); ++i)
			{
				std::size_t input_index = i;

				for (std::size_t j = 0; j < count; ++j)
				{
					this->output()[i] += base_->output()[input_index];
					input_index += block_size_;
				}

				this->output()[i] /= static_cast<scalar_type>(count);
			}
		}

		virtual void backward() override
		{
			BOOST_ASSERT(base_->output().size() % block_size_ == 0);
			std::size_t count = base_->output().size() / block_size_;
			scalar_type inv_count = static_cast<scalar_type>(1.0) / static_cast<scalar_type>(count);

#pragma omp parallel for schedule(static)
			for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(block_size_); ++i)
			{
				std::size_t input_index = i;

				for (std::size_t j = 0; j < count; ++j)
				{
					base_->output_grad()[input_index] += this->output_grad()[i] * inv_count;
					input_index += block_size_;
				}
			}
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(base_.get());
		}

	private:

		node_pointer base_;
		std::size_t block_size_;

	};

	template <class T>
	expression<T> depth_mean(const expression<T>& base, std::size_t block_size)
	{
		return expression<T>(std::make_shared<depth_mean_expression_node<T>>(base.root(), block_size));
	}

}
