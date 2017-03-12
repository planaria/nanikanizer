#pragma once
#include "optimizer_impl_holder_base.hpp"
#include "variable.hpp"

namespace nnk
{

	class optimizer_base : boost::noncopyable
	{
	public:

		virtual ~optimizer_base()
		{
		}

		template <class layer_type>
		void add_parameter(layer_type& l)
		{
			l.enumerate_parameters(*this);
		}

		template <class T>
		void add_parameter(variable<T>& param)
		{
			holders_.push_back(create_holder(param));
		}

		void zero_grads()
		{
			#pragma omp parallel for schedule(dynamic)
			for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(holders_.size()); ++i)
				holders_[i]->zero_grads();
		}

		void update()
		{
			#pragma omp parallel for schedule(dynamic)
			for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(holders_.size()); ++i)
				holders_[i]->update();
		}

	protected:

		virtual std::unique_ptr<optimizer_impl_holder_base>
			create_holder(variable<float>& variable) const = 0;

		virtual std::unique_ptr<optimizer_impl_holder_base>
			create_holder(variable<double>& variable) const = 0;

	private:

		std::vector<std::unique_ptr<optimizer_impl_holder_base>> holders_;

	};

}
