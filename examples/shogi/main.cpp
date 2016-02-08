#include <iostream>
#include <iomanip>
#include <nanikanizer/nanikanizer.hpp>
#include "game.hpp"
#include "enumerate_action.hpp"
#include "apply_state.hpp"
#include "random_state_generator.hpp"

int main(int /*argc*/, char* /*argv*/[])
{
	using namespace shogi;

	try
	{
		random_state_generator gen;

		nnk::variable<float> input;
		nnk::linear_layer<float> l1(input_size, 1500);

		auto x1 = input.expr();
		auto x2 = nnk::sigmoid(l1.forward(x1));

		{
			nnk::linear_layer<float> l1_backward(1500, input_size);

			auto x1_ = nnk::sigmoid(l1_backward.forward(x2));
			auto loss = nnk::cross_entropy(x1_ - x1);

			nnk::adam_optimizer optimizer;
			optimizer.add_parameter(l1);
			optimizer.add_parameter(l1_backward);

			nnk::evaluator<float> ev(loss);

			std::size_t batch_size = 100;
			input.value().resize(input_size * batch_size);

			while (true)
			{
				optimizer.zero_grads();

				gen.generate(&input.value()[0], batch_size);

				std::cout << ev.forward()[0] << std::endl;
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
