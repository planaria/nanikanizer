#pragma once
#include "optimizer_impl_holder_base.hpp"
#include "optimizer_impl_base.hpp"

namespace nnk
{

	template <class T>
	class optimizer_impl_holder : public optimizer_impl_holder_base
	{
	public:

		typedef T scalar_type;
		typedef std::valarray<scalar_type> tensor_type;
		typedef variable<scalar_type> variable_type;
		typedef optimizer_impl_base<scalar_type> optimizer_impl_type;
		typedef std::unique_ptr<optimizer_impl_type> optimizer_impl_ptr;

		optimizer_impl_holder(variable_type& variable, optimizer_impl_ptr optimizer)
			: variable_(variable)
			, optimizer_(std::move(optimizer))
		{
		}

		virtual void zero_grads() override
		{
			variable_.zero_grads();
		}

		virtual void update() override
		{
			optimizer_->update(variable_.value(), variable_.grad());
		}

	private:

		variable_type variable_;
		optimizer_impl_ptr optimizer_;

	};

}
