#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("max_pooling_2d")
{
	nnk::variable<double> x =
	{
		11.0, 12.0, 13.0, 14.0,
		21.0, 22.0, 23.0, 24.0,
		31.0, 32.0, 33.0, 34.0,
		41.0, 42.0, 43.0, 44.0,
	};

	auto y = max_pooling_2d(x.expr(), 4, 4, 1, 2, 2);

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 4);
	CHECK(result[0] == Approx(22.0));
	CHECK(result[1] == Approx(24.0));
	CHECK(result[2] == Approx(42.0));
	CHECK(result[3] == Approx(44.0));

	ev.backward({ 1.0, 1.0, 1.0, 1.0 });

	REQUIRE(x.node()->output_grad().size() == 16);
	CHECK(x.node()->output_grad()[0] == Approx(0.0));
	CHECK(x.node()->output_grad()[1] == Approx(0.0));
	CHECK(x.node()->output_grad()[2] == Approx(0.0));
	CHECK(x.node()->output_grad()[3] == Approx(0.0));
	CHECK(x.node()->output_grad()[4] == Approx(0.0));
	CHECK(x.node()->output_grad()[5] == Approx(1.0));
	CHECK(x.node()->output_grad()[6] == Approx(0.0));
	CHECK(x.node()->output_grad()[7] == Approx(1.0));
	CHECK(x.node()->output_grad()[8] == Approx(0.0));
	CHECK(x.node()->output_grad()[9] == Approx(0.0));
	CHECK(x.node()->output_grad()[10] == Approx(0.0));
	CHECK(x.node()->output_grad()[11] == Approx(0.0));
	CHECK(x.node()->output_grad()[12] == Approx(0.0));
	CHECK(x.node()->output_grad()[13] == Approx(1.0));
	CHECK(x.node()->output_grad()[14] == Approx(0.0));
	CHECK(x.node()->output_grad()[15] == Approx(1.0));
}

TEST_CASE("sum_pooling_2d")
{
	nnk::variable<double> x =
	{
		11.0, 12.0, 13.0, 14.0,
		21.0, 22.0, 23.0, 24.0,
		31.0, 32.0, 33.0, 34.0,
		41.0, 42.0, 43.0, 44.0,
	};

	auto y = sum_pooling_2d(x.expr(), 4, 4, 1, 2, 2);

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 4);
	CHECK(result[0] == Approx(66.0));
	CHECK(result[1] == Approx(74.0));
	CHECK(result[2] == Approx(146.0));
	CHECK(result[3] == Approx(154.0));

	ev.backward({ 1.0, 1.0, 1.0, 1.0 });

	REQUIRE(x.node()->output_grad().size() == 16);
	CHECK(x.node()->output_grad()[0] == Approx(1.0));
	CHECK(x.node()->output_grad()[1] == Approx(1.0));
	CHECK(x.node()->output_grad()[2] == Approx(1.0));
	CHECK(x.node()->output_grad()[3] == Approx(1.0));
	CHECK(x.node()->output_grad()[4] == Approx(1.0));
	CHECK(x.node()->output_grad()[5] == Approx(1.0));
	CHECK(x.node()->output_grad()[6] == Approx(1.0));
	CHECK(x.node()->output_grad()[7] == Approx(1.0));
	CHECK(x.node()->output_grad()[8] == Approx(1.0));
	CHECK(x.node()->output_grad()[9] == Approx(1.0));
	CHECK(x.node()->output_grad()[10] == Approx(1.0));
	CHECK(x.node()->output_grad()[11] == Approx(1.0));
	CHECK(x.node()->output_grad()[12] == Approx(1.0));
	CHECK(x.node()->output_grad()[13] == Approx(1.0));
	CHECK(x.node()->output_grad()[14] == Approx(1.0));
	CHECK(x.node()->output_grad()[15] == Approx(1.0));
}
