#pragma once
#include "optimizer_adapter.hpp"
#include "math_util.hpp"

namespace nnk
{

	class adadelta_optimizer : public optimizer_adapter<adadelta_optimizer>
	{
	public:

		explicit adadelta_optimizer(
			double gamma = 0.999,
			double eps = 1.0e-8)
			: gamma_(gamma)
			, eps_(eps)
		{
		}

		template <class T>
		std::unique_ptr<optimizer_impl_base<T>> create_impl() const
		{
			return std::make_unique<impl<T>>(
				static_cast<T>(gamma_),
				static_cast<T>(eps_));
		}

	private:

		template <class T>
		class impl : public optimizer_impl_base<T>
		{
		public:

			explicit impl(
				scalar_type gamma,
				scalar_type eps)
				: gamma_(gamma)
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

				r_ = lerp(norm_sq(grad), r_, gamma_);

				for (std::size_t i = 0; i < grad.size(); ++i)
					v_[i] = (std::sqrt(s_) + eps_) / (std::sqrt(r_) + eps_) * grad[i];

				s_ = lerp(norm_sq(v_), s_, gamma_);
				x -= v_;
			}

		private:

			bool initialized_ = false;

			scalar_type gamma_;
			scalar_type eps_;

			scalar_type r_ = 0.0;
			tensor_type v_;
			scalar_type s_ = 0;

		};

		double gamma_;
		double eps_;

	};

}
