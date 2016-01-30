#pragma once
#include "expression.hpp"
#include "variable_expression.hpp"

namespace nnk
{

	template <class T>
	class variable
	{
	public:

		typedef T scalar_type;
		typedef std::valarray<scalar_type> tensor_type;
		typedef variable_expression_node<scalar_type> node_type;
		typedef std::shared_ptr<node_type> node_pointer;

		variable()
			: node_(std::make_shared<node_type>())
		{
		}

		variable(const tensor_type& value)
			: node_(std::make_shared<node_type>(value))
		{
		}

		variable(const std::initializer_list<scalar_type>& value)
			: node_(std::make_shared<node_type>(value))
		{
		}

		variable(scalar_type value)
			: node_(std::make_shared<node_type>(value))
		{
		}

		explicit variable(std::size_t size)
			: node_(std::make_shared<node_type>(size))
		{
		}

		const node_pointer& node() const
		{
			return node_;
		}

		const tensor_type& value() const
		{
			return node_->output();
		}

		tensor_type& value()
		{
			return node_->output();
		}

		expression<scalar_type> expr() const
		{
			return expression<scalar_type>(node_);
		}

	private:

		std::shared_ptr<node_type> node_;

	};

}
