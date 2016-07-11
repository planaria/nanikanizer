#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("depth_mean")
{
	nnk::variable<double> x =
	{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	};

	auto y = nnk::depth_mean(x.expr(), 3);

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 3);
	CHECK(result[0] == Approx(2.5));
	CHECK(result[1] == Approx(3.5));
	CHECK(result[2] == Approx(4.5));

	ev.backward(
	{
		1.0, 1.0, 1.0,
	});

	REQUIRE(x.grad().size() == 6);
	CHECK(x.grad()[0] == Approx(0.5));
	CHECK(x.grad()[1] == Approx(0.5));
	CHECK(x.grad()[2] == Approx(0.5));
	CHECK(x.grad()[3] == Approx(0.5));
	CHECK(x.grad()[4] == Approx(0.5));
	CHECK(x.grad()[5] == Approx(0.5));
}
