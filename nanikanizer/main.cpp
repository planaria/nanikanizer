#include <iostream>
#include <iomanip>
#include <nanikanizer/nanikanizer.hpp>

int main(int /*argc*/, char* /*argv*/[])
{
	try
	{
		std::vector<std::pair<std::valarray<double>, std::valarray<double>>> data =
		{
			{ { 0.0, 0.0, 1.0 },{ 0.0, 1.0 } },
			{ { 0.0, 0.0, 0.5 },{ 0.0, 0.75 } },
			{ { 0.0, 1.0, 0.0 },{ 1.0, 0.5 } },
			{ { 1.0, 0.0, 0.0 },{ 1.0, 0.5 } },
			{ { 0.5, 0.5, 1.0 },{ 1.0, 1.0 } },
			{ { 1.0, 1.0, 1.0 },{ 2.0, 1.0 } },
		};

		nnk::variable<double> x;
		nnk::variable<double> y;

		nnk::linear_layer<double> layer(3, 2);

		auto loss = norm_sq(layer(x) - y);

		nnk::adam_optimizer optimizer;
		optimizer.add_parameter(layer);

		nnk::evaluator<double> ev(loss);

		std::cout << std::fixed << std::setprecision(8);

		while (true)
		{
			optimizer.zero_grads();

			for (const auto& d : data)
			{
				x.value() = d.first;
				y.value() = d.second;

				std::cout << ev.forward()[0] << " ";
				ev.backward();
			}

			optimizer.update();

			std::cout << std::endl;
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
