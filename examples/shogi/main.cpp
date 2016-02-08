#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <nanikanizer/nanikanizer.hpp>
#include "game.hpp"
#include "enumerate_action.hpp"
#include "apply_state.hpp"
#include "random_state_generator.hpp"

int main(int /*argc*/, char* /*argv*/[])
{
	namespace fs = boost::filesystem;

	try
	{
		shogi::random_state_generator gen;

		nnk::variable<float> input;

		auto pretrain = [&](
			nnk::linear_layer<float>& layer,
			nnk::expression<float>& prev,
			nnk::expression<float>& next,
			std::size_t batch_size,
			std::size_t repeat,
			const fs::path& cache)
		{
			fs::ofstream log(fs::change_extension(cache, ".log"));

			if (fs::exists(cache))
			{
				fs::ifstream is(cache, std::ios_base::binary);
				nnk::binary_reader reader(is);
				layer.load(reader);
			}
			else
			{
				nnk::linear_layer<float> layer_backward(layer.output_dimension(), layer.input_dimension());

				auto prev_ = nnk::sigmoid(layer_backward.forward(next));
				auto loss = nnk::cross_entropy(prev_ - prev);

				nnk::adam_optimizer optimizer;
				optimizer.add_parameter(layer);
				optimizer.add_parameter(layer_backward);

				nnk::evaluator<float> ev(loss);

				input.value().resize(shogi::input_size * batch_size);

				for (std::size_t i = 0; i < repeat; ++i)
				{
					optimizer.zero_grads();

					gen.generate(&input.value()[0], batch_size);

					log << i << "," << ev.forward()[0] << std::endl;
					ev.backward();

					optimizer.update();
				}

				fs::ofstream os(cache, std::ios_base::binary);
				nnk::binary_writer writer(os);
				layer.save(writer);
			}
		};

		nnk::linear_layer<float> l1(shogi::input_size, 1500);
		nnk::linear_layer<float> l2(1500, 1000);
		nnk::linear_layer<float> l3(1000, 750);
		nnk::linear_layer<float> l4(750, 500);
		nnk::linear_layer<float> l5(500, 250);

		auto x1 = input.expr();
		auto x2 = nnk::sigmoid(l1.forward(x1));
		auto x3 = nnk::sigmoid(l2.forward(x2));
		auto x4 = nnk::sigmoid(l3.forward(x3));
		auto x5 = nnk::sigmoid(l4.forward(x4));
		auto x6 = nnk::sigmoid(l5.forward(x5));

		pretrain(l1, x1, x2, 100, 10000, "layer1.nan");
		pretrain(l2, x2, x3, 100, 10000, "layer2.nan");
		pretrain(l3, x3, x4, 100, 10000, "layer3.nan");
		pretrain(l4, x4, x5, 100, 10000, "layer4.nan");
		pretrain(l5, x5, x6, 100, 10000, "layer5.nan");
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
