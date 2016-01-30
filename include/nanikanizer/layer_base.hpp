#pragma once
#include "optimizer_base.hpp"

namespace nnk
{

	template <class T>
	class layer_base : boost::noncopyable
	{
	public:
		
		typedef T scalar_type;
		typedef std::valarray<scalar_type> tensor_type;

		virtual ~layer_base()
		{
		}

		virtual void enumerate_parameters(optimizer_base& optimizer) = 0;

		virtual expression<scalar_type> forward(const expression<scalar_type>& v) const = 0;

	};

}
