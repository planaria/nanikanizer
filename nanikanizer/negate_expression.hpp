#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class negate_expression_node : public expression_node<T>
	{
	public:

		negate_expression_node(const node_pointer& base)
			: base_(base)
		{
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			if (output().size() != base_->output().size())
				output().resize(base_->output().size());

			for (std::size_t i = 0; i < base_->output().size(); ++i)
				output()[i] = -base_->output()[i];
		}

		virtual void backward() override
		{
			for (std::size_t i = 0; i < output_grad().size(); ++i)
				base_->output_grad()[i] -= output_grad()[i];
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(base_.get());
		}

	private:

		node_pointer base_;

	};

}
