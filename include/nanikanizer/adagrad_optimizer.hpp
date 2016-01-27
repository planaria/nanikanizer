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
		private:

			typedef optimizer_impl_base<T> base_type;

		public:

			typedef typename base_type::scalar_type scalar_type;
			typedef typename base_type::tensor_type tensor_type;

			explicit impl(
				scalar_type alpha,
				scalar_type eps)
				: alpha_(alpha)
				, eps_(eps)
			{
			}

			virtual void update(tensor_type& x, const tensor_type& grad) override
			{
				if (!initialized_)
				{
					r_.resize(x.size(), static_cast<scalar_type>(0.0));
					initialized_ = true;
				}

				for (std::size_t i = 0; i < grad.size(); ++i)
				{
					r_[i] += square(grad[i]);
					x[i] -= alpha_ / (std::sqrt(r_[i]) + eps_) * grad[i];
				}
			}

		private:

			bool initialized_ = false;

			scalar_type alpha_;
			scalar_type eps_;

			tensor_type r_;

		};

		double alpha_;
		double eps_;

	};

}
