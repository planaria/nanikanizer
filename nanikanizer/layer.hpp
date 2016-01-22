#pragma once
#include "optimizer_base.hpp"

namespace nnk
{

	class layer
	{
	public:

		virtual ~layer()
		{
		}

		virtual void enumerate_parameters(optimizer_base& optimizer) = 0;

	};

}
