#pragma once
#include "expression.hpp"
#include "expression_node.hpp"

namespace nnk
{

	template <class T>
	class dropout_expression_node : public expression_node<T>
	{
	private:

		typedef expression_node<T> base_type;

	public:

		typedef typename base_type::scalar_type scalar_type;
		typedef typename base_type::tensor_type tensor_type;
		typedef typename base_type::node_pointer node_pointer;

		dropout_expression_node(const node_pointer& base, scalar_type ratio, const std::shared_ptr<bool>& train)
			: base_(base)
			, ratio_(ratio)
			, train_(train)
		{
			BOOST_ASSERT(train);
		}

		virtual bool is_branch() override
		{
			return true;
		}

		virtual void forward() override
		{
			if (this->output().size() != base_->output().size())
				this->output().resize(base_->output().size());

			if (*train_)
			{
				if (filter_.size() != base_->output().size())
					filter_.resize(base_->output().size());

				std::uniform_real_distribution<> random;

				for (std::size_t i = 0; i < filter_.size(); ++i)
					filter_.set(i, ratio_ < random(generator_));

				for (std::size_t i = 0; i < base_->output().size(); ++i)
				{
					if (filter_.test(i))
						this->output()[i] = base_->output()[i];
					else
						this->output()[i] = static_cast<scalar_type>(0.0);
				}
			}
			else
			{
				for (std::size_t i = 0; i < base_->output().size(); ++i)
					this->output()[i] = base_->output()[i] * ratio_;
			}
		}

		virtual void backward() override
		{
			if (*train_)
			{
				BOOST_ASSERT(filter_.size() == this->output_grad().size());

				for (std::size_t i = 0; i < this->output_grad().size(); ++i)
				{
					if (filter_.test(i))
						base_->output_grad()[i] += this->output_grad()[i];
				}
			}
			else
			{
				for (std::size_t i = 0; i < this->output_grad().size(); ++i)
					base_->output_grad()[i] += this->output_grad()[i] * ratio_;
			}
		}

		virtual void enumerate_children(const std::function<void(expression_node_base*)>& callback) override
		{
			callback(base_.get());
		}

	private:

		node_pointer base_;

		scalar_type ratio_;
		std::shared_ptr<bool> train_;

		std::mt19937 generator_;

		boost::dynamic_bitset<> filter_;

	};

	template <class T>
	expression<T> dropout(const expression<T>& base, T ratio, const std::shared_ptr<bool>& train)
	{
		return expression<T>(std::make_shared<dropout_expression_node<T>>(base.root(), ratio, train));
	}

}
