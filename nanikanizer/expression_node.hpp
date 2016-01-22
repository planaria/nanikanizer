#pragma once
#include "expression_node_base.hpp"

namespace nnk
{

	template <class T>
	class expression_node : public expression_node_base
	{
	public:

		typedef T scalar_type;
		typedef std::valarray<scalar_type> tensor_type;
		typedef std::shared_ptr<expression_node<scalar_type>> node_pointer;

		const tensor_type& output() const
		{
			return output_;
		}

		tensor_type& output()
		{
			return output_;
		}

		const tensor_type& output_grad() const
		{
			BOOST_ASSERT(output_grad_.size() == output_.size());
			return output_grad_;
		}

		tensor_type& output_grad()
		{
			BOOST_ASSERT(output_grad_.size() == output_.size());
			return output_grad_;
		}

		virtual void zero_grads() override
		{
			output_grad_ = static_cast<scalar_type>(0.0);
		}

		virtual void prepare_grads() override
		{
			if (output_grad_.size() != output_.size())
				output_grad_.resize(output_.size(), static_cast<scalar_type>(0.0));
		}

	private:

		tensor_type output_;
		tensor_type output_grad_;

	};

}
