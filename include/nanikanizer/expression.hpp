#pragma once
#include "expression_node.hpp"
#include "constant_expression.hpp"

namespace nnk
{

	template <class T>
	class expression
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
			: root_(std::make_shared<constant_expression_node<scalar_type>>(value))
		{
		}

		expression(const std::initializer_list<scalar_type>& value)
			: root_(std::make_shared<constant_expression_node<scalar_type>>(value))
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

	private:

		static const node_pointer& zero_node()
		{
			static node_pointer node = std::make_shared<constant_expression_node<T>>(std::valarray<scalar_type>{ scalar_type() });
			return node;
		}

		node_pointer root_;

	};

	template <class T>
	bool operator ==(const expression<T>& lhs, const expression<T>& rhs)
	{
		return lhs.root() == rhs.root();
	}

	template <class T>
	bool operator !=(const expression<T>& lhs, const expression<T>& rhs)
	{
		return lhs.root() != rhs.root();
	}

}

namespace std
{

	template <class T>
	struct hash<nnk::expression<T>>
	{
		std::size_t operator ()(const nnk::expression<T>& e)
		{
			return hash<typename nnk::expression<T>::node_pointer>()(e.root());
		}
	};

}
