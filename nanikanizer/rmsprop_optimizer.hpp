#pragma once
#include "optimizer_adapter.hpp"
#include "math_util.hpp"

namespace nnk
{

	class rmsprop_optimizer : public optimizer_adapter<rmsprop_optimizer>
	{
	public:

		explicit rmsprop_optimizer(
			double alpha = 0.001,
			double gamma = 0.999,
			double eps = 1.0e-8)
			: alpha_(alpha)
			, gamma_(gamma)
			, eps_(eps)
		{
		}

		template <class T>
		std::unique_ptr<optimizer_impl_base<T>> create_impl() const
		{
			return std::make_unique<impl<T>>(
				static_cast<T>(alpha_),
				static_cast<T>(gamma_),
				static_cast<T>(eps_));
		}

	private:

		template <class T>
		class impl : public optimizer_impl_base<T>
		{
		private:

			typedef optimizer_impl_base<T> base_type;

		public:

			typedef typename base_type::scalar_type scalar_type;
			typedef typename base_type::tensor_type tensor_type;

			explicit impl(
				scalar_type alpha,
				scalar_type gamma,
				scalar_type eps)
				: alpha_(alpha)
				, gamma_(gamma)
				, eps_(eps)
			{
			}

			virtual void update(tensor_type& x, const tensor_type& grad) override
			{
				r_ = lerp(norm_sq(grad), r_, gamma_);

				for (std::size_t i = 0; i < grad.size(); ++i)
					x[i] -= alpha_ / (std::sqrt(r_) + eps_) * grad[i];
			}

		private:

			scalar_type alpha_;
			scalar_type gamma_;
			scalar_type eps_;

			scalar_type r_ = 0.0;

		};

		double alpha_;
		double gamma_;
		double eps_;

	};

}
