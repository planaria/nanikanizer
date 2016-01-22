#pragma once

namespace nnk
{

	class optimizer_base;

	class layer
	{
	public:

		virtual ~layer()
		{
		}

		virtual void enumerate_parameters(optimizer_base& optimizer) = 0;

	};

}
