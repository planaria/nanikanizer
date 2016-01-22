#pragma once
#include "optimizer_adapter.hpp"
#include "math_util.hpp"

namespace nnk
{

	class adagrad_optimizer : public optimizer_adapter<adagrad_optimizer>
	{
	public:

		explicit adagrad_optimizer(
			double alpha = 0.001,
			double eps = 1.0e-8)
			: alpha_(alpha)
			, eps_(eps)
		{
		}

		template <class T>
		std::unique_ptr<optimizer_impl_base<T>> create_impl() const
		{
			return std::make_unique<impl<T>>(
				static_cast<T>(alpha_),
				static_cast<T>(eps_));
		}

	private:

		template <class T>
		class impl : public optimizer_impl_base<T>
		{
		public:

			explicit impl(
				scalar_type alpha,
				scalar_type eps)
				: alpha_(alpha)
				, eps_(eps)
			{
			}

			virtual void update(tensor_type& x, const tensor_type& grad) override
			{
				r_ += norm_sq(grad);

				for (std::size_t i = 0; i < grad.size(); ++i)
					x[i] -= alpha_ / (std::sqrt(r_) + eps_) * grad[i];
			}

		private:

			scalar_type alpha_;
			scalar_type eps_;

			scalar_type r_ = 0.0;

		};

		double alpha_;
		double eps_;

	};

}
