#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class sum_expression_node : public expression_node<T>
	{
	public:

		explicit sum_expression_node(const node_pointer& base)
			: base_(base)
		{
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			output() = { base_->output().sum() };
		}

		virtual void backward() override
		{
			BOOST_ASSERT(output_grad().size() == 1);
			base_->output_grad() += output_grad()[0];
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(base_.get());
		}

	private:

		node_pointer base_;

	};

	template <class T>
	expression<T> sum(const expression<T>& base)
	{
		return expression<T>(std::make_shared<sum_expression_node<T>>(base.root()));
	}

}
