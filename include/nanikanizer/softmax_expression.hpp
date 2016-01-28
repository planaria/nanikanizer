#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class softmax_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		explicit softmax_expression_node(const node_pointer& base, const boost::optional<std::size_t>& block_size)
			: base_(base)
			, block_size_(block_size ? *block_size : 0)
			, enable_block_(block_size)
		{
			if (block_size)
				BOOST_ASSERT(*block_size >= 2);
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			std::size_t block_size = enable_block_ ? block_size_ : base_->output().size();

			BOOST_ASSERT(base_->output().size() % block_size == 0);
			std::size_t count = base_->output().size() / block_size;

			if (this->output().size() != base_->output().size())
				this->output().resize(base_->output().size());

			std::size_t base_index = 0;

			for (std::size_t i = 0; i < count; ++i)
			{
				scalar_type max = -std::numeric_limits<scalar_type>::max();

				for (std::size_t j = 0; j < block_size; ++j)
					max = std::max(max, base_->output()[base_index + j]);

				scalar_type sum = scalar_type();

				for (std::size_t j = 0; j < block_size; ++j)
				{
					scalar_type t = std::exp(base_->output()[base_index + j] - max);
					this->output()[base_index + j] = t;
					sum += t;
				}

				for (std::size_t j = 0; j < block_size; ++j)
					this->output()[base_index + j] /= sum;

				base_index += block_size;
			}
		}

		virtual void backward() override
		{
			std::size_t block_size = enable_block_ ? block_size_ : base_->output().size();

			if (temp_.size() != block_size)
				temp_.resize(block_size);

			BOOST_ASSERT(base_->output().size() % block_size == 0);
			std::size_t count = base_->output().size() / block_size;

			std::size_t base_index = 0;

			for (std::size_t i = 0; i < count; ++i)
			{
				scalar_type sum = scalar_type();

				for (std::size_t j = 0; j < block_size; ++j)
				{
					scalar_type t = this->output()[base_index + j] * this->output_grad()[base_index + j];
					temp_[j] = t;
					sum += t;
				}

				for (std::size_t j = 0; j < block_size; ++j)
					base_->output_grad()[base_index + j] += temp_[j] - this->output()[base_index + j] * sum;

				base_index += block_size;
			}
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(base_.get());
		}

	private:

		node_pointer base_;

		std::size_t block_size_;
		bool enable_block_;

		tensor_type temp_;

	};

	template <class T>
	expression<T> softmax(const expression<T>& base, const boost::optional<std::size_t>& block_size = boost::none)
	{
		return expression<T>(std::make_shared<softmax_expression_node<T>>(base.root(), block_size));
	}

}
