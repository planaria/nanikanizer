#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T, class OP>
	class binary_operator_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		binary_operator_expression_node(const node_pointer& lhs, const node_pointer& rhs)
			: lhs_(lhs)
			, rhs_(rhs)
		{
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			if (lhs_->output().size() < rhs_->output().size())
			{
				if (this->output().size() != rhs_->output().size())
					this->output().resize(rhs_->output().size());

				BOOST_ASSERT(rhs_->output().size() % lhs_->output().size() == 0);
				std::size_t count = rhs_->output().size() / lhs_->output().size();

				std::size_t output_index = 0;

				for (std::size_t i = 0; i < count; ++i)
				{
					for (std::size_t j = 0; j < lhs_->output().size(); ++j)
					{
						this->output()[output_index] = OP::forward(lhs_->output()[j], rhs_->output()[output_index]);
						++output_index;
					}
				}
			}
			else if (lhs_->output().size() > rhs_->output().size())
			{
				if (this->output().size() != lhs_->output().size())
					this->output().resize(lhs_->output().size());

				BOOST_ASSERT(lhs_->output().size() % rhs_->output().size() == 0);
				std::size_t count = lhs_->output().size() / rhs_->output().size();

				std::size_t output_index = 0;

				for (std::size_t i = 0; i < count; ++i)
				{
					for (std::size_t j = 0; j < rhs_->output().size(); ++j)
					{
						this->output()[output_index] = OP::forward(lhs_->output()[output_index], rhs_->output()[j]);
						++output_index;
					}
				}
			}
			else
			{
				if (this->output().size() != lhs_->output().size())
					this->output().resize(lhs_->output().size());

				for (std::size_t i = 0; i < lhs_->output().size(); ++i)
					this->output()[i] = OP::forward(lhs_->output()[i], rhs_->output()[i]);
			}
		}

		virtual void backward() override
		{
			if (lhs_->output().size() < rhs_->output().size())
			{
				BOOST_ASSERT(rhs_->output().size() % lhs_->output().size() == 0);
				std::size_t count = rhs_->output().size() / lhs_->output().size();

				std::size_t output_index = 0;

				for (std::size_t i = 0; i < count; ++i)
				{
					for (std::size_t j = 0; j < lhs_->output().size(); ++j)
					{
						OP::backward(
							lhs_->output_grad()[j],
							rhs_->output_grad()[output_index],
							lhs_->output()[j],
							rhs_->output()[output_index],
							this->output()[output_index],
							this->output_grad()[output_index]);

						++output_index;
					}
				}
			}
			else if (lhs_->output().size() > rhs_->output().size())
			{
				BOOST_ASSERT(lhs_->output().size() % rhs_->output().size() == 0);
				std::size_t count = lhs_->output().size() / rhs_->output().size();

				std::size_t output_index = 0;

				for (std::size_t i = 0; i < count; ++i)
				{
					for (std::size_t j = 0; j < rhs_->output().size(); ++j)
					{
						OP::backward(
							lhs_->output_grad()[output_index],
							rhs_->output_grad()[j],
							lhs_->output()[output_index],
							rhs_->output()[j],
							this->output()[output_index],
							this->output_grad()[output_index]);

						++output_index;
					}
				}
			}
			else
			{
				for (std::size_t i = 0; i < lhs_->output().size(); ++i)
				{
					OP::backward(
						lhs_->output_grad()[i],
						rhs_->output_grad()[i],
						lhs_->output()[i],
						rhs_->output()[i],
						this->output()[i],
						this->output_grad()[i]);
				}
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

	};

	template <class OP, class T>
	expression<T> binary_operator(const expression<T>& lhs, const expression<T>& rhs)
	{
		return expression<T>(std::make_shared<binary_operator_expression_node<T, OP>>(lhs.root(), rhs.root()));
	}

}
