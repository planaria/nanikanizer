#pragma once
#include "optimizer_base.hpp"
#include "binary_writer.hpp"
#include "binary_reader.hpp"

namespace nnk
{

	class layer_base : boost::noncopyable
	{
	public:
		
		virtual ~layer_base()
		{
		}

		virtual void save(binary_writer& writer) const = 0;

		virtual void load(binary_reader& reader) = 0;

		virtual void enumerate_parameters(optimizer_base& optimizer) = 0;

	};

}
