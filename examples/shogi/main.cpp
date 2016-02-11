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

	namespace fs = boost::filesystem;

	class shogi_layers : boost::noncopyable
	{
	public:

		shogi_layers()
			: l1_(input_size, 1500)
			, l2_(1500, 1000)
			, l3_(1000, 750)
			, l4_(750, 500)
			, l5_(500, 250)
			, l6_(250, 1)
		{
		}

		void pre_train()
		{
			nnk::variable<float> input;

			auto x1 = input.expr();
			auto x2 = nnk::sigmoid(l1_.forward(x1));
			auto x3 = nnk::sigmoid(l2_.forward(x2));
			auto x4 = nnk::sigmoid(l3_.forward(x3));
			auto x5 = nnk::sigmoid(l4_.forward(x4));
			auto x6 = nnk::sigmoid(l5_.forward(x5));

			pre_train_layer(input, l1_, x1, x2, 100, 10000, "layer1.nan");
			pre_train_layer(input, l2_, x2, x3, 100, 10000, "layer2.nan");
			pre_train_layer(input, l3_, x3, x4, 100, 10000, "layer3.nan");
			pre_train_layer(input, l4_, x4, x5, 100, 10000, "layer4.nan");
			pre_train_layer(input, l5_, x5, x6, 100, 10000, "layer5.nan");
		}

		void fine_tune()
		{
			fs::path cache = "all.nnk";

			if (fs::exists(cache))
			{
				fs::ifstream is(cache, std::ios_base::binary);
				nnk::binary_reader reader(is);
				l1_.load(reader);
				l2_.load(reader);
				l3_.load(reader);
				l4_.load(reader);
				l5_.load(reader);
				l6_.load(reader);
			}
			else
			{
				nnk::variable<float> input;
				nnk::variable<float> input_next;

				auto output = forward_all(input.expr());
				auto output_next = nnk::min(forward_all(input_next.expr()));

				auto loss_normal = nnk::abs(output + output_next);

				auto scale = nnk::expression<float>({ 5.0f });
				auto loss_win = scale * nnk::abs(output - nnk::expression<float>{ 1.0f });

				nnk::evaluator<float> ev_normal(loss_normal);
				nnk::evaluator<float> ev_win(loss_win);

				nnk::adam_optimizer optimizer(0.0001);
				optimizer.add_parameter(l1_);
				optimizer.add_parameter(l2_);
				optimizer.add_parameter(l3_);
				optimizer.add_parameter(l4_);
				optimizer.add_parameter(l5_);
				optimizer.add_parameter(l6_);

				random_state_generator gen;

				std::vector<action> actions;

				while (true)
				{
					optimizer.zero_grads();

					for (std::size_t i = 0; i < 100; ++i)
					{
						gen.randomize();

						const game& g = gen.get();

						input.value().resize(input_size);
						apply_state(&input.value()[0], g.state(), g.turn());

						actions.clear();
						enumerate_action(g, std::back_inserter(actions));

						input_next.value().resize(input_size * actions.size());

						bool win = false;

						for (std::size_t j = 0; j < actions.size(); ++j)
						{
							const action& action = actions[j];

							game next_game = g;
							next_game.apply(action);

							if (next_game.state().is_finished())
								win = true;

							float* p = &input_next.value()[input_size * j];
							apply_state(p, next_game.state(), next_game.turn());
						}

						//std::cout << g.state() << std::endl;

						nnk::evaluator<float>& ev = win ? ev_win : ev_normal;

						std::cout << win << ":" << ev.forward()[0] << std::endl;
						ev.backward();
					}

					optimizer.update();

					{
						fs::ofstream os(cache, std::ios_base::binary);
						nnk::binary_writer writer(os);
						l1_.save(writer);
						l2_.save(writer);
						l3_.save(writer);
						l4_.save(writer);
						l5_.save(writer);
						l6_.save(writer);
					}

					std::cout << "----------" << std::endl;
				}
			}
		}

		float evaluate(const shogi::game& g)
		{
			nnk::variable<float> input;
			auto output = forward_all(input.expr());

			input.value().resize(input_size);
			apply_state(&input.value()[0], g.state(), g.turn());

			nnk::evaluator<float> ev(output);
			ev.forward();

			return output.root()->output()[0];
		}

	private:

		void pre_train_layer(
			nnk::variable<float>& input,
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

				input.value().resize(input_size * batch_size);

				for (std::size_t i = 0; i < repeat; ++i)
				{
					optimizer.zero_grads();

					gen_.generate(&input.value()[0], batch_size);

					log << ev.forward()[0] << std::endl;
					ev.backward();

					optimizer.update();
				}

				fs::ofstream os(cache, std::ios_base::binary);
				nnk::binary_writer writer(os);
				layer.save(writer);
			}
		}

		nnk::expression<float> forward_all(const nnk::expression<float>& x1)
		{
			auto x2 = nnk::sigmoid(l1_.forward(x1));
			auto x3 = nnk::sigmoid(l2_.forward(x2));
			auto x4 = nnk::sigmoid(l3_.forward(x3));
			auto x5 = nnk::sigmoid(l4_.forward(x4));
			auto x6 = nnk::sigmoid(l5_.forward(x5));
			auto x7 = nnk::tanh(l6_.forward(x6));
			return x7;
		}

		nnk::linear_layer<float> l1_;
		nnk::linear_layer<float> l2_;
		nnk::linear_layer<float> l3_;
		nnk::linear_layer<float> l4_;
		nnk::linear_layer<float> l5_;
		nnk::linear_layer<float> l6_;

		random_state_generator gen_;

	};

}

int main(int /*argc*/, char* /*argv*/[])
{
	try
	{
		shogi::shogi_layers layers;

		layers.pre_train();

		layers.fine_tune();

		shogi::game g;

		std::vector<shogi::action> actions;
		std::vector<float> action_values;
		
		while (!g.state().is_finished())
		{
			std::cout << g.state() << std::endl;

			actions.clear();
			enumerate_action(g, std::back_inserter(actions));

			action_values.resize(actions.size());

			std::transform(actions.begin(), actions.end(), action_values.begin(),
				[&](const shogi::action& action)
			{
				shogi::game next_game = g;
				next_game.apply(action);

				if (next_game.state().is_finished())
					return FLT_MAX;

				return -layers.evaluate(next_game);
			});

			auto it = std::max_element(action_values.begin(), action_values.end());
			std::size_t index = it - action_values.begin();

			g.apply(actions[index]);
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
