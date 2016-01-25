#pragma once

namespace nnk
{

	class optimizer_impl_holder_base : boost::noncopyable
	{
	public:

		virtual ~optimizer_impl_holder_base()
		{
		}

		virtual void zero_grads() = 0;

		virtual void update() = 0;

	};

}
