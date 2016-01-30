#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("linear_layer")
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

	auto loss = norm_sq(layer.forward(x.expr()) - y.expr());

	nnk::adam_optimizer optimizer;
	optimizer.add_parameter(layer);

	nnk::evaluator<double> ev(loss);

	for (std::size_t i = 0; i < 10000; ++i)
	{
		optimizer.zero_grads();

		for (const auto& d : data)
		{
			x.value() = d.first;
			y.value() = d.second;

			ev.forward();
			ev.backward();
		}

		optimizer.update();
	}

	for (const auto& d : data)
	{
		x.value() = d.first;
		y.value() = d.second;

		auto result = ev.forward();

		REQUIRE(result.size() == 1);
		CHECK(result[0] == Approx(0.0));
	}
}
