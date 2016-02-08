#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("abs")
{
	nnk::variable<double> x = { 2.0, -3.0 };

	auto y = abs(x.expr());

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 2);
	CHECK(result[0] == Approx(2.0));
	CHECK(result[1] == Approx(3.0));

	ev.backward({ 1.0, 1.0 });

	REQUIRE(x.grad().size() == 2);
	CHECK(x.grad()[0] == Approx(1.0));
	CHECK(x.grad()[1] == Approx(-1.0));
}

TEST_CASE("square")
{
	nnk::variable<double> x = { 3.0 };

	auto y = square(x.expr());

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(9.0));

	ev.backward();

	REQUIRE(x.grad().size() == 1);
	CHECK(x.grad()[0] == Approx(6.0));
}

TEST_CASE("sqrt")
{
	nnk::variable<double> x = { 9.0 };

	auto y = sqrt(x.expr());

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(3.0));

	ev.backward();

	REQUIRE(x.grad().size() == 1);
	CHECK(x.grad()[0] == Approx(1.0 / 6.0));
}
