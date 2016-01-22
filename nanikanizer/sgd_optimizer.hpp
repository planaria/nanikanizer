#pragma once
#include "optimizer_adapter.hpp"
#include "math_util.hpp"

namespace nnk
{

	class sgd_optimizer : public optimizer_adapter<sgd_optimizer>
	{
	public:

		explicit sgd_optimizer(double alpha = 0.001)
			: alpha_(alpha)
		{
		}

		template <class T>
		std::unique_ptr<optimizer_impl_base<T>> create_impl() const
		{
			return std::make_unique<impl<T>>(
				static_cast<T>(alpha_));
		}

	private:

		template <class T>
		class impl : public optimizer_impl_base<T>
		{
		public:

			explicit impl(scalar_type alpha)
				: alpha_(alpha)
			{
			}

			virtual void update(tensor_type& x, const tensor_type& grad) override
			{
				for (std::size_t i = 0; i < grad.size(); ++i)
					x[i] -= alpha_ * grad[i];
			}

		private:

			scalar_type alpha_;

		};

		double alpha_;

	};

}
