#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class sum_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		explicit sum_expression_node(const node_pointer& base, const boost::optional<std::size_t>& block_size)
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
			if (enable_block_)
			{
				BOOST_ASSERT(base_->output().size() % block_size_ == 0);
				std::size_t count = base_->output().size() / block_size_;

				if (this->output().size() != count)
					this->output().resize(count);

				std::size_t base_index = 0;

				for (std::size_t i = 0; i < count; ++i)
				{
					this->output()[i] = static_cast<scalar_type>(0.0);

					for (std::size_t j = 0; j < block_size_; ++j)
					{
						this->output()[i] += base_->output()[base_index];
						++base_index;
					}
				}
			}
			else
			{
				this->output() = { base_->output().sum() };
			}
		}

		virtual void backward() override
		{
			if (enable_block_)
			{
				BOOST_ASSERT(base_->output().size() % block_size_ == 0);
				std::size_t count = base_->output().size() / block_size_;

				std::size_t base_index = 0;

				for (std::size_t i = 0; i < count; ++i)
				{
					for (std::size_t j = 0; j < block_size_; ++j)
					{
						base_->output_grad()[base_index] += this->output_grad()[i];
						++base_index;
					}
				}
			}
			else
			{
				base_->output_grad() += this->output_grad()[0];
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

	};

	template <class T>
	expression<T> sum(const expression<T>& base, const boost::optional<std::size_t>& block_size = boost::none)
	{
		return expression<T>(std::make_shared<sum_expression_node<T>>(base.root(), block_size));
	}

}
