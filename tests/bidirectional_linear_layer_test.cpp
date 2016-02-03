#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("bidirectional_linear_layer")
{
	nnk::expression<double> x =
	{
		1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
	};

	nnk::bidirectional_linear_layer<double> layer(6, 5);

	auto loss = norm_sq(layer.backward(layer.forward(x)) - x);

	nnk::adam_optimizer optimizer;
	optimizer.add_parameter(layer);

	nnk::evaluator<double> ev(loss);

	for (std::size_t i = 0; i < 10000; ++i)
	{
		optimizer.zero_grads();
		ev.forward();
		ev.backward();
		optimizer.update();
	}

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(0.0));
}
