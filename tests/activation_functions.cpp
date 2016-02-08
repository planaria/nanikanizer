#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("sigmoid")
{
	nnk::variable<double> x = { 0.0 };

	auto y = sigmoid(x.expr());

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(0.5));

	ev.backward();

	REQUIRE(x.grad().size() == 1);
	CHECK(x.grad()[0] == Approx(0.25));
}

TEST_CASE("tanh")
{
	nnk::variable<double> x = { std::atanh(0.5) };

	auto y = tanh(x.expr());

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(0.5));

	ev.backward();

	REQUIRE(x.grad().size() == 1);
	CHECK(x.grad()[0] == Approx(0.75));
}

TEST_CASE("relu")
{
	nnk::variable<double> x = { -1.0, 1.0 };

	auto y = relu(x.expr());

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 2);
	CHECK(result[0] == Approx(0.0));
	CHECK(result[1] == Approx(1.0));

	ev.backward({ 1.0, 1.0 });

	REQUIRE(x.grad().size() == 2);
	CHECK(x.grad()[0] == Approx(0.0));
	CHECK(x.grad()[1] == Approx(1.0));
}
