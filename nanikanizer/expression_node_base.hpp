#pragma once

namespace nnk
{

	class expression_node_base : boost::noncopyable
	{
	public:

		virtual ~expression_node_base()
		{
		}

		virtual bool is_branch() = 0;

		virtual void forward() = 0;

		virtual void backward() = 0;

		virtual void zero_grads() = 0;

		virtual void prepare_grads() = 0;

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) = 0;

	};

}
