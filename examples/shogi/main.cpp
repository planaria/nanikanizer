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

	class shogi_ai
	{
	public:

		shogi_ai()
			: l1_(input_size, 1500)
			, l2_(1500, 1000)
			, l3_(1000, 750)
			, l4_(750, 500)
			, l5_(500, 250)
			, l6_(250, 1)
			, opt_(0.0001)
		{
			opt_.add_parameter(l1_);
			opt_.add_parameter(l2_);
			opt_.add_parameter(l3_);
			opt_.add_parameter(l4_);
			opt_.add_parameter(l5_);
			opt_.add_parameter(l6_);
		}

		float evaluate(const game& g)
		{
			auto e = forward(g);
			nnk::evaluator<float> ev(e);
			ev.forward();

			return e.root()->output()[0];
		}

		action step(const game& g)
		{
			BOOST_ASSERT(!g.state().is_finished());

			std::vector<action> actions;
			enumerate_action(g, std::back_inserter(actions));

			std::vector<float> values(actions.size());

			#pragma omp parallel for schedule(dynamic)
			for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(actions.size()); ++i)
			{
				auto next_g = g;
				next_g.apply(actions[i]);

				values[i] = -evaluate(next_g);
			}

			std::size_t maximum_index = std::max_element(values.begin(), values.end()) - values.begin();
			return actions[maximum_index];
		}

		void zero_grads()
		{
			opt_.zero_grads();
		}

		float train(const game& g1, const game& g2)
		{
			auto e1 = forward(g1);
			auto e2 = forward(g2);

			auto loss = nnk::square(e1 + e2);
			nnk::evaluator<float> ev(loss);
			ev.forward();
			ev.backward();

			return loss.root()->output()[0];
		}

		void update()
		{
			opt_.update();
		}

		void save(const boost::filesystem::path& filename)
		{
			std::ofstream os;
			os.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
			os.open(filename.c_str(), std::ios::binary);

			nnk::binary_writer writer(os);
			l1_.save(writer);
			l2_.save(writer);
			l3_.save(writer);
			l4_.save(writer);
			l5_.save(writer);
			l6_.save(writer);
		}

		void load(const boost::filesystem::path& filename)
		{
			std::ifstream is;
			is.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
			is.open(filename.c_str(), std::ios::binary);

			nnk::binary_reader reader(is);
			l1_.load(reader);
			l2_.load(reader);
			l3_.load(reader);
			l4_.load(reader);
			l5_.load(reader);
			l6_.load(reader);
		}

	private:

		nnk::expression<float> forward(const game& g)
		{
			if (g.state().is_finished())
			{
				if (g.turn() ^ g.state().is_win())
					return { 1.0 };
				else
					return { -1.0 };
			}

			nnk::variable<float> input;

			input.value().resize(input_size);
			apply_state(&input.value()[0], g.state(), g.turn());

			auto e = input.expr();
			e = nnk::relu(l1_.forward(e));
			e = nnk::relu(l2_.forward(e));
			e = nnk::relu(l3_.forward(e));
			e = nnk::relu(l4_.forward(e));
			e = nnk::relu(l5_.forward(e));
			e = nnk::tanh(l6_.forward(e));

			return e;
		}

		nnk::linear_layer<float> l1_;
		nnk::linear_layer<float> l2_;
		nnk::linear_layer<float> l3_;
		nnk::linear_layer<float> l4_;
		nnk::linear_layer<float> l5_;
		nnk::linear_layer<float> l6_;

		nnk::adam_optimizer opt_;

	};

}

int main(int /*argc*/, char* /*argv*/[])
{
	try
	{
		std::cout << std::fixed << std::setprecision(3);

		shogi::shogi_ai ai;

		boost::filesystem::path filename = "shogi.dat";

		if (boost::filesystem::exists(filename))
			ai.load(filename);

		shogi::random_state_generator gen;

		while (true)
		{
			gen.randomize();

			shogi::game g = gen.get();

			std::vector<shogi::game> states;

			for (std::size_t i = 0; i < 1000; ++i)
			{
				float value = ai.evaluate(g);

				if (g.turn())
					std::cout << value;

				std::cout << std::endl;

				std::cout << g.state();

				if (!g.turn())
					std::cout << value;

				std::cout << std::endl;

				states.push_back(g);

				if (g.state().is_finished())
					break;

				auto next_g = g;
				next_g.apply(ai.step(g));

				g = next_g;
			}

			float sum = 0.0f;

			ai.zero_grads();

			for (std::size_t i = 0; i < states.size() - 1; ++i)
				sum += ai.train(states[i], states[i + 1]);

			ai.update();

			std::cout << "loss: " << sum << std::endl;

			ai.save(filename);
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
