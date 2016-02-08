#pragma once
#include "optimizer_base.hpp"
#include "optimizer_impl_holder.hpp"

namespace nnk
{

	template <class T>
	class optimizer_adapter : public optimizer_base
	{
	protected:

		typedef T derived_type;

		virtual std::unique_ptr<optimizer_impl_holder_base>
			create_holder(variable<float>& variable) const override
		{
			return create_holder_impl(variable);
		}

		virtual std::unique_ptr<optimizer_impl_holder_base>
			create_holder(variable<double>& variable) const override
		{
			return create_holder_impl(variable);
		}

	private:

		template <class U>
		std::unique_ptr<optimizer_impl_holder_base>
			create_holder_impl(variable<U>& variable) const
		{
			return std::make_unique<optimizer_impl_holder<U>>(variable, derived().template create_impl<U>());;
		}

		const derived_type& derived() const
		{
			return static_cast<const derived_type&>(*this);
		}

	};

}
