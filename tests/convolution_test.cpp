#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("convolution")
{
	nnk::variable<double> x =
	{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	};

	auto y = nnk::convolution_2d(x.expr(), 2, 3, 1, 2, 2);

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 8);
	CHECK(result[0] == Approx(1.0));
	CHECK(result[1] == Approx(2.0));
	CHECK(result[2] == Approx(4.0));
	CHECK(result[3] == Approx(5.0));
	CHECK(result[4] == Approx(2.0));
	CHECK(result[5] == Approx(3.0));
	CHECK(result[6] == Approx(5.0));
	CHECK(result[7] == Approx(6.0));

	ev.backward(
	{
		1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0,
	});

	REQUIRE(x.grad().size() == 6);
	CHECK(x.grad()[0] == Approx(1.0));
	CHECK(x.grad()[1] == Approx(2.0));
	CHECK(x.grad()[2] == Approx(1.0));
	CHECK(x.grad()[3] == Approx(1.0));
	CHECK(x.grad()[4] == Approx(2.0));
	CHECK(x.grad()[5] == Approx(1.0));
}
