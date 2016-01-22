#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class constant_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base;

	public:

		typedef typename base::scalar_type scalar_type;
		typedef typename base::tensor_type tensor_type;
		typedef typename base::node_pointer node_pointer;

		constant_expression_node(const tensor_type& value)
		{
			this->output() = value;
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

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& /*callback*/) override
		{
		}

	};

}
