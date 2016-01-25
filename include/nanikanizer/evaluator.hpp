#pragma once
#include "expression.hpp"

namespace nnk
{

	template <class T>
	class evaluator : boost::noncopyable
	{
	public:

		typedef T scalar_type;
		typedef std::valarray<scalar_type> tensor_type;
		typedef expression<scalar_type> expression_type;
		typedef typename expression_type::node_type node_type;

		explicit evaluator(expression_type& expr)
			: expr_(expr)
		{
			std::unordered_map<expression_node_base*, std::size_t> vertex_map;
			std::vector<expression_node_base*> vertices;

			enumerate_vertices(expr.root().get(),
				[&](expression_node_base* node)
			{
				if (!vertex_map.insert(std::make_pair(node, vertex_map.size())).second)
					return false;

				vertices.push_back(node);
				return true;
			});

			boost::adjacency_list<> g;

			for (std::size_t i = 0; i < vertices.size(); ++i)
				boost::add_vertex(g);

			for (const auto& key_value : vertex_map)
			{
				key_value.first->enumerate_children(
					[&](expression_node_base* child)
				{
					boost::add_edge(vertex_map[child], key_value.second, g);
				});
			}

			nodes_.reserve(vertices.size());
			branch_nodes_.reserve(vertices.size());

			boost::topological_sort(g, boost::make_function_output_iterator(
				[&](std::size_t index)
			{
				expression_node_base* node = vertices[index];

				nodes_.push_back(node);

				if (node->is_branch())
					branch_nodes_.push_back(node);
			}));
		}

		const tensor_type& forward()
		{
			for (expression_node_base* node : boost::adaptors::reverse(nodes_))
			{
				node->forward();
				node->prepare_grads();
			}

			return expr_.root()->output();
		}

		void backward(const tensor_type& initial_grad = one())
		{
			BOOST_ASSERT(expr_.root()->output().size() == initial_grad.size());
			expr_.root()->output_grad() = initial_grad;

			for (auto it = std::next(branch_nodes_.begin()); it != branch_nodes_.end(); ++it)
				(*it)->zero_grads();

			for (expression_node_base* node : branch_nodes_)
				node->backward();
		}

	private:

		static const tensor_type& one()
		{
			static tensor_type result = { 1.0 };
			return result;
		}

		static void enumerate_vertices(
			expression_node_base* node,
			const std::function<bool (expression_node_base*)>& callback)
		{
			if (!callback(node))
				return;

			node->enumerate_children(
				[&](expression_node_base* child)
			{
				enumerate_vertices(child, callback);
			});
		}

		expression_type expr_;
		std::vector<expression_node_base*> nodes_;
		std::vector<expression_node_base*> branch_nodes_;

	};

}
