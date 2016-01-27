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
		private:

			typedef optimizer_impl_base<T> base_type;

		public:

			typedef typename base_type::scalar_type scalar_type;
			typedef typename base_type::tensor_type tensor_type;

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
					r_.resize(x.size(), static_cast<scalar_type>(0.0));
					v_.resize(x.size(), static_cast<scalar_type>(0.0));
					s_.resize(x.size(), static_cast<scalar_type>(0.0));
					initialized_ = true;
				}

				for (std::size_t i = 0; i < grad.size(); ++i)
				{
					r_[i] = lerp(square(grad[i]), r_[i], gamma_);
					v_[i] = (std::sqrt(s_[i]) + eps_) / (std::sqrt(r_[i]) + eps_) * grad[i];
					s_[i] = lerp(square(v_[i]), s_[i], gamma_);
				}

				x -= v_;
			}

		private:

			bool initialized_ = false;

			scalar_type gamma_;
			scalar_type eps_;

			tensor_type r_;
			tensor_type v_;
			tensor_type s_;

		};

		double gamma_;
		double eps_;

	};

}
