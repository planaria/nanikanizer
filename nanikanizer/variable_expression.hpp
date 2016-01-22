#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class variable_expression_node : public expression_node<T>
	{
	public:

		variable_expression_node()
		{
		}

		variable_expression_node(const tensor_type& value)
		{
			output() = value;
		}

		virtual bool is_branch() override
		{
			return false;
		}

		virtual void forward() override
		{
		}

		virtual void backward() override
		{
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
		}

	};

}
