#pragma once
#include "layer_base.hpp"

namespace nnk
{

	template <class T>
	class dropout_layer : public layer_base
	{
	public:

		typedef T scalar_type;
		typedef std::valarray<scalar_type> tensor_type;

		explicit dropout_layer(scalar_type ratio = 0.5)
			: ratio_(ratio)
		{
		}

		virtual void save(binary_writer& writer) const override
		{
			writer.write(ratio_);
			writer.write(*train_);
		}

		virtual void load(binary_reader& reader) override
		{
			reader.read(ratio_);
			reader.read(*train_);
		}

		virtual void enumerate_parameters(optimizer_base& /*optimizer*/) override
		{
		}

		double ratio() const
		{
			return ratio_;
		}

		double& ratio()
		{
			return ratio_;
		}

		bool train() const
		{
			return *train_;
		}

		bool& train()
		{
			return *train_;
		}

		expression<scalar_type> forward(const expression<scalar_type>& v) const
		{
			return dropout(v, ratio_, train_);
		}

	private:

		scalar_type ratio_;
		std::shared_ptr<bool> train_ = std::make_shared<bool>(true);

	};

}
