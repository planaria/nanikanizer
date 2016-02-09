#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T, class OP>
	class block_operator_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		explicit block_operator_expression_node(const node_pointer& base, const boost::optional<std::size_t>& block_size)
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
					this->output()[i] = OP::forward(&base_->output()[base_index], block_size_);
					base_index += block_size_;
				}
			}
			else
			{
				if (this->output().size() != 1)
					this->output().resize(1);

				this->output()[0] = OP::forward(&base_->output()[0], base_->output().size());
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
					OP::backward(
						&base_->output_grad()[base_index],
						&base_->output()[base_index],
						block_size_,
						this->output()[i],
						this->output_grad()[i]);

					base_index += block_size_;
				}
			}
			else
			{
				OP::backward(
					&base_->output_grad()[0],
					&base_->output()[0],
					base_->output().size(),
					this->output()[0],
					this->output_grad()[0]);
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

	template <class OP, class T>
	expression<T> block_operator(const expression<T>& base, const boost::optional<std::size_t>& block_size = boost::none)
	{
		return expression<T>(std::make_shared<block_operator_expression_node<T, OP>>(base.root(), block_size));
	}

}
