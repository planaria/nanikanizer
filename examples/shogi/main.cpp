#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <nanikanizer/nanikanizer.hpp>
#include "game.hpp"
#include "enumerate_action.hpp"
#include "apply_state.hpp"
#include "random_state_generator.hpp"

namespace shogi
{

	void pretrain(
		nnk::linear_layer<float>& l1,
		nnk::linear_layer<float>& l2,
		nnk::linear_layer<float>& l3,
		nnk::linear_layer<float>& l4,
		nnk::linear_layer<float>& l5)
	{
		namespace fs = boost::filesystem;

		shogi::random_state_generator gen;

		nnk::variable<float> input;

		auto pretrain_layer = [&](
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

					log << ev.forward()[0] << std::endl;
					ev.backward();

					optimizer.update();
				}

				fs::ofstream os(cache, std::ios_base::binary);
				nnk::binary_writer writer(os);
				layer.save(writer);
			}
		};

		auto x1 = input.expr();
		auto x2 = nnk::sigmoid(l1.forward(x1));
		auto x3 = nnk::sigmoid(l2.forward(x2));
		auto x4 = nnk::sigmoid(l3.forward(x3));
		auto x5 = nnk::sigmoid(l4.forward(x4));
		auto x6 = nnk::sigmoid(l5.forward(x5));

		pretrain_layer(l1, x1, x2, 100, 10000, "layer1.nan");
		pretrain_layer(l2, x2, x3, 100, 10000, "layer2.nan");
		pretrain_layer(l3, x3, x4, 100, 10000, "layer3.nan");
		pretrain_layer(l4, x4, x5, 100, 10000, "layer4.nan");
		pretrain_layer(l5, x5, x6, 100, 10000, "layer5.nan");
	}

}

int main(int /*argc*/, char* /*argv*/[])
{
	namespace fs = boost::filesystem;

	try
	{
		nnk::linear_layer<float> l1(shogi::input_size, 1500);
		nnk::linear_layer<float> l2(1500, 1000);
		nnk::linear_layer<float> l3(1000, 750);
		nnk::linear_layer<float> l4(750, 500);
		nnk::linear_layer<float> l5(500, 250);

		shogi::pretrain(l1, l2, l3, l4, l5);

		nnk::linear_layer<float> l6(250, 1);

		auto forward_all = [&](const nnk::expression<float>& x1)
		{
			auto x2 = nnk::sigmoid(l1.forward(x1));
			auto x3 = nnk::sigmoid(l2.forward(x2));
			auto x4 = nnk::sigmoid(l3.forward(x3));
			auto x5 = nnk::sigmoid(l4.forward(x4));
			auto x6 = nnk::sigmoid(l5.forward(x5));
			auto x7 = l6.forward(x6);
			return x7;
		};

		nnk::variable<float> input;
		nnk::variable<float> input_next;

		auto output = forward_all(input.expr());
		auto output_next = nnk::min(forward_all(input_next.expr()));

		auto loss_normal = nnk::abs(output + output_next);
		auto loss_win = nnk::abs(output - nnk::expression<float>{ 1.0 });

		nnk::evaluator<float> ev_normal(loss_normal);
		nnk::evaluator<float> ev_win(loss_win);

		nnk::adam_optimizer optimizer;
		optimizer.add_parameter(l1);
		optimizer.add_parameter(l2);
		optimizer.add_parameter(l3);
		optimizer.add_parameter(l4);
		optimizer.add_parameter(l5);
		optimizer.add_parameter(l6);

		shogi::random_state_generator gen;

		std::vector<shogi::action> actions;

		while (true)
		{
			gen.randomize();

			const shogi::game& g = gen.get();

			input.value().resize(shogi::input_size);
			shogi::apply_state(&input.value()[0], g.state(), g.turn());

			actions.clear();
			shogi::enumerate_action(g, std::back_inserter(actions));

			input_next.value().resize(shogi::input_size * actions.size());

			bool win = false;

			for (std::size_t j = 0; j < actions.size(); ++j)
			{
				const shogi::action& action = actions[j];

				shogi::game next_game = g;
				next_game.apply(action);

				if (next_game.state().is_finished())
				{
					win = true;
					break;
				}

				float* p = &input_next.value()[shogi::input_size * j];
				shogi::apply_state(p, next_game.state(), next_game.turn());
			}

			//std::cout << g.state() << std::endl;

			nnk::evaluator<float>& ev = win ? ev_win : ev_normal;
			std::size_t count = win ? 100 : 1;

			for (std::size_t i = 0; i < count; ++i)
			{
				optimizer.zero_grads();

				std::cout << win << ":" << ev.forward()[0] << std::endl;
				ev.backward();

				optimizer.update();
			}
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
