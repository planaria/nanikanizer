#pragma once
#include "expression_node.hpp"
#include "constant_expression.hpp"
#include "negate_expression.hpp"
#include "add_expression.hpp"
#include "subtract_expression.hpp"
#include "multiply_expression.hpp"
#include "divide_expression.hpp"

namespace nnk
{

	template <class T>
	class expression
		: boost::addable<expression<T>
		, boost::subtractable<expression<T>
		, boost::multipliable<expression<T>
		, boost::dividable<expression<T>
		>>>>
	{
	public:
		
		typedef T scalar_type;
		typedef std::valarray<scalar_type> tensor_type;
		typedef expression_node<scalar_type> node_type;
		typedef std::shared_ptr<node_type> node_pointer;

		expression()
			: root_(zero_node())
		{
		}

		expression(const node_pointer& node)
			: root_(node)
		{
		}

		expression(const tensor_type& value)
			: root_(std::make_shared<constant_expression_node<tensor_type>>(value))
		{
		}

		const node_pointer& root() const
		{
			return root_;
		}

		const expression& operator +()
		{
			return *this;
		}

		expression operator -() const
		{
			return expression(std::make_shared<negate_expression_node<scalar_type>>(root_));
		}

		expression& operator +=(const expression& rhs)
		{
			root_ = std::make_shared<add_expression_node<scalar_type>>(root_, rhs.root_);
			return *this;
		}

		expression& operator -=(const expression& rhs)
		{
			root_ = std::make_shared<subtract_expression_node<scalar_type>>(root_, rhs.root_);
			return *this;
		}

		expression& operator *=(const expression& rhs)
		{
			root_ = std::make_shared<multiply_expression_node<scalar_type>>(root_, rhs.root());
			return *this;
		}

		expression& operator /=(const expression& rhs)
		{
			root_ = std::make_shared<divide_expression_node<scalar_type>>(root_, rhs.root_);
			return *this;
		}

	private:

		static const node_pointer& zero_node()
		{
			static node_pointer node = std::make_shared<constant_expression_node<T>>();
			return node;
		}

		node_pointer root_;

	};

}
