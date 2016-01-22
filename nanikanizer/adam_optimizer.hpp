#pragma once
#include "optimizer_adapter.hpp"
#include "math_util.hpp"

namespace nnk
{

	class adam_optimizer : public optimizer_adapter<adam_optimizer>
	{
	public:

		adam_optimizer(
			double alpha = 0.001,
			double beta1 = 0.9,
			double beta2 = 0.999,
			double eps = 1.0e-8)
			: alpha_(alpha)
			, beta1_(beta1)
			, beta2_(beta2)
			, eps_(eps)
		{
		}

		template <class T>
		std::unique_ptr<optimizer_impl_base<T>> create_impl() const
		{
			return std::make_unique<impl<T>>(
				static_cast<T>(alpha_),
				static_cast<T>(beta1_),
				static_cast<T>(beta2_),
				static_cast<T>(eps_));
		}

	private:

		template <class T>
		class impl : public optimizer_impl_base<T>
		{
		private:

			typedef optimizer_impl_base<T> base;

		public:

			typedef typename base::scalar_type scalar_type;
			typedef typename base::tensor_type tensor_type;

			impl(
				scalar_type alpha,
				scalar_type beta1,
				scalar_type beta2,
				scalar_type eps)
				: alpha_(alpha)
				, beta1_(beta1)
				, beta2_(beta2)
				, eps_(eps)
			{
			}

			virtual void update(tensor_type& x, const tensor_type& grad) override
			{
				if (!initialized_)
				{
					v_.resize(x.size(), static_cast<scalar_type>(0.0));
					initialized_ = true;
				}

				beta1t_ *= beta1_;
				beta2t_ *= beta2_;

				for (std::size_t i = 0; i < grad.size(); ++i)
					v_[i] = lerp(grad[i], v_[i], beta1_);

				r_ = lerp(norm_sq(grad), r_, beta2_);

				scalar_type v_ratio = static_cast<scalar_type>(1.0) - beta1t_;
				scalar_type r_ratio = static_cast<scalar_type>(1.0) - beta2t_;
				scalar_type scale = alpha_ / v_ratio / (std::sqrt(r_ / r_ratio) + eps_);

				for (std::size_t i = 0; i < grad.size(); ++i)
					x[i] -= v_[i] * scale;
			}

		private:

			bool initialized_ = false;

			scalar_type alpha_;
			scalar_type beta1_;
			scalar_type beta2_;
			scalar_type eps_;

			tensor_type v_;
			scalar_type r_ = static_cast<scalar_type>(0.0);
			scalar_type beta1t_ = static_cast<scalar_type>(1.0);
			scalar_type beta2t_ = static_cast<scalar_type>(1.0);

		};

		double alpha_;
		double beta1_;
		double beta2_;
		double eps_;

	};

}
