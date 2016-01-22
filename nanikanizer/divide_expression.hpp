#pragma once
#include "expression.hpp"
#include "expression_node.hpp"
#include "math_util.hpp"

namespace nnk
{

	template <class T>
	class divide_expression_node : public expression_node<T>
	{
	public:

		typedef expression_node<T> node_type;
		typedef std::shared_ptr<node_type> node_pointer;

		typedef expression_node<scalar_type> scalar_node_type;
		typedef std::shared_ptr<scalar_node_type> scalar_node_pointer;

		divide_expression_node(const node_pointer& lhs, const scalar_node_pointer& rhs)
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
			if (lhs_->output().size() == 1)
			{
				if (output().size() != rhs_->output().size())
					output().resize(rhs_->output().size());

				for (std::size_t i = 0; i < rhs_->output().size(); ++i)
					output()[i] = lhs_->output()[0] / rhs_->output()[i];
			}
			else if (rhs_->output().size() == 1)
			{
				if (output().size() != lhs_->output().size())
					output().resize(lhs_->output().size());

				for (std::size_t i = 0; i < lhs_->output().size(); ++i)
					output()[i] = lhs_->output()[i] / rhs_->output()[0];
			}
			else
			{
				BOOST_ASSERT(lhs_->output().size() == rhs_->output().size());

				if (output().size() != lhs_->output().size())
					output().resize(lhs_->output().size());

				for (std::size_t i = 0; i < lhs_->output().size(); ++i)
					output()[i] = lhs_->output()[i] / rhs_->output()[i];
			}
		}

		virtual void backward() override
		{
			if (lhs_->output().size() == 1)
			{
				for (std::size_t i = 0; i < output_grad().size(); ++i)
				{
					lhs_->output_grad()[0] += output_grad()[i] / rhs_->output()[i];
					rhs_->output_grad()[i] += output_grad()[i] * lhs_->output()[0] / pow2(rhs_->output()[i]);
				}
			}
			else if (rhs_->output().size() == 1)
			{
				for (std::size_t i = 0; i < output_grad().size(); ++i)
				{
					lhs_->output_grad()[i] += output_grad()[i] / rhs_->output()[0];
					rhs_->output_grad()[0] += output_grad()[i] * lhs_->output()[i] / pow2(rhs_->output()[0]);
				}
			}
			else
			{
				for (std::size_t i = 0; i < output_grad().size(); ++i)
				{
					lhs_->output_grad()[i] += output_grad()[i] / rhs_->output()[i];
					rhs_->output_grad()[i] += output_grad()[i] * lhs_->output()[i] / pow2(rhs_->output()[i]);
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
		scalar_node_pointer rhs_;

	};

}
