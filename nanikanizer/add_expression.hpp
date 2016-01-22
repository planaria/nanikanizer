#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class add_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base;

	public:

		typedef typename base::scalar_type scalar_type;
		typedef typename base::tensor_type tensor_type;
		typedef typename base::node_pointer node_pointer;

		add_expression_node(const node_pointer& lhs, const node_pointer& rhs)
			: lhs_(lhs)
			, rhs_(rhs)
		{
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			if (lhs_->output().size() == 1)
			{
				if (this->output().size() != rhs_->output().size())
					this->output().resize(rhs_->output().size());

				for (std::size_t i = 0; i < rhs_->output().size(); ++i)
					this->output()[i] = lhs_->output()[0] + rhs_->output()[i];
			}
			else if (rhs_->output().size() == 1)
			{
				if (this->output().size() != lhs_->output().size())
					this->output().resize(lhs_->output().size());

				for (std::size_t i = 0; i < lhs_->output().size(); ++i)
					this->output()[i] = lhs_->output()[i] + rhs_->output()[0];
			}
			else
			{
				BOOST_ASSERT(lhs_->output().size() == rhs_->output().size());

				if (this->output().size() != lhs_->output().size())
					this->output().resize(lhs_->output().size());

				for (std::size_t i = 0; i < lhs_->output().size(); ++i)
					this->output()[i] = lhs_->output()[i] + rhs_->output()[i];
			}
		}

		virtual void backward() override
		{
			if (lhs_->output().size() == 1)
			{
				lhs_->output_grad() += this->output_grad().sum();
				rhs_->output_grad() += this->output_grad();
			}
			else if (rhs_->output().size() == 1)
			{
				lhs_->output_grad() += this->output_grad();
				rhs_->output_grad() += this->output_grad().sum();
			}
			else
			{
				lhs_->output_grad() += this->output_grad();
				rhs_->output_grad() += this->output_grad();
			}
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(lhs_.get());
			callback(rhs_.get());
		}

	private:

		node_pointer lhs_;
		node_pointer rhs_;

	};

}
