#include <iostream>
#include <iomanip>
#include <nanikanizer/nanikanizer.hpp>

int main(int /*argc*/, char* /*argv*/[])
{
	try
	{
		nnk::expression<double> x =
		{
			0.0,
			1.0,
			2.0,
			3.0,
			4.0,
			5.0,
		};

		nnk::expression<double> y =
		{
			1.0,
			-4.0,
			-5.0,
			-2.0,
			5.0,
			16.0,
		};

		nnk::variable<double> a = { 0.0 };
		nnk::variable<double> b = { 0.0 };
		nnk::variable<double> c = { 0.0 };

		auto fx = a.expr() * x * x + b.expr() * x + c.expr();
		auto loss = nnk::norm_sq(fx - y);

		nnk::evaluator<double> ev(loss);

		nnk::adam_optimizer optimizer(0.1);
		optimizer.add_parameter(a);
		optimizer.add_parameter(b);
		optimizer.add_parameter(c);

		for (std::size_t i = 0; i < 1500; ++i)
		{
			optimizer.zero_grads();

			ev.forward();
			ev.backward();

			optimizer.update();

			std::cout << std::fixed << std::setprecision(8);
			std::cout
				<< "a = " << a.value()[0] << ", "
				<< "b = " << b.value()[0] << ", "
				<< "c = " << c.value()[0] << std::endl;
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
