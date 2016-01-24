#pragma once
#include "optimizer_base.hpp"

namespace nnk
{

	class layer_base : boost::noncopyable
	{
	public:

		virtual ~layer_base()
		{
		}

		virtual void enumerate_parameters(optimizer_base& optimizer) = 0;

	};

}
