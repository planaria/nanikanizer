#pragma once

namespace nnk
{

	template <class T>
	class optimizer_impl_base : boost::noncopyable
	{
	public:

		typedef T scalar_type;
		typedef std::valarray<scalar_type> tensor_type;

		virtual ~optimizer_impl_base()
		{
		}

		virtual void update(tensor_type& x, const tensor_type& grad) = 0;

	};

}
