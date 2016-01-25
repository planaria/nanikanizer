#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class depth_concat_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		explicit depth_concat_expression_node(std::vector<node_pointer> nodes)
			: nodes_(std::move(nodes))
		{
			BOOST_ASSERT(!nodes_.empty());
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			BOOST_ASSERT(std::all_of(nodes_.begin(), nodes_.end(),
				[&](const node_pointer& node)
			{
				return node->output().size() == nodes_.front()->output().size();
			}));

			std::size_t output_size = nodes_.front()->output().size();
			std::size_t total_output_size = output_size * nodes_.size();

			if (this->output().size() != total_output_size)
				this->output().resize(total_output_size);

			std::size_t output_index = 0;

			for (std::size_t i = 0; i < output_size; ++i)
			{
				for (std::size_t j = 0; j < nodes_.size(); ++j)
				{
					this->output()[output_index] = nodes_[j]->output()[i];
					++output_index;
				}
			}
		}

		virtual void backward() override
		{
			std::size_t output_size = nodes_.front()->output().size();

			std::size_t output_index = 0;

			for (std::size_t i = 0; i < output_size; ++i)
			{
				for (std::size_t j = 0; j < nodes_.size(); ++j)
				{
					nodes_[j]->output_grad()[i] += this->output_grad()[output_index];
					++output_index;
				}
			}
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			for (const node_pointer& node : nodes_)
				callback(node.get());
		}

	private:

		std::vector<node_pointer> nodes_;

	};

	template <class T>
	expression<T> depth_concat(const std::initializer_list<expression<T>>& expressions)
	{
		typedef expression_node<T> node_type;
		typedef std::shared_ptr<node_type> node_pointer;

		std::vector<node_pointer> nodes(expressions.size());

		std::transform(expressions.begin(), expressions.end(), nodes.begin(),
			[](const expression<T>& expression)
		{
			return expression.root();
		});

		return expression<T>(std::make_shared<depth_concat_expression_node<T>>(std::move(nodes)));
	}

}
