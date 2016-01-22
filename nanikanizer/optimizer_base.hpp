#pragma once
#include "layer.hpp"
#include "optimizer_impl_holder_base.hpp"

namespace nnk
{

	class optimizer_base : boost::noncopyable
	{
	public:

		virtual ~optimizer_base()
		{
		}

		void add_parameter(layer& l)
		{
			l.enumerate_parameters(*this);
		}

		template <class T>
		void add_parameter(variable<T>& param)
		{
			holders_.push_back(create_holder(*param.node()));
		}

		void zero_grads()
		{
			for (const auto& holder : holders_)
				holder->zero_grads();
		}

		void update()
		{
			for (const auto& holder : holders_)
				holder->update();
		}

	protected:

		virtual std::unique_ptr<optimizer_impl_holder_base>
			create_holder(variable_expression_node<float>& variable) const = 0;

		virtual std::unique_ptr<optimizer_impl_holder_base>
			create_holder(variable_expression_node<double>& variable) const = 0;

	private:

		std::vector<std::unique_ptr<optimizer_impl_holder_base>> holders_;

	};

}
