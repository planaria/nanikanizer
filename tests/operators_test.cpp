#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("negate")
{
	nnk::variable<double> x = { 2.0 };

	auto y = -x.expr();

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(-2.0));

	ev.backward();

	REQUIRE(x.node()->output_grad().size() == 1);
	CHECK(x.node()->output_grad()[0] == Approx(-1.0));
}

TEST_CASE("add")
{
	nnk::variable<double> x1 = { 1.0 };
	nnk::variable<double> x2 = { 2.0 };

	auto y = x1.expr() + x2.expr();

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(3.0));

	ev.backward();

	REQUIRE(x1.node()->output_grad().size() == 1);
	CHECK(x1.node()->output_grad()[0] == Approx(1.0));

	REQUIRE(x2.node()->output_grad().size() == 1);
	CHECK(x2.node()->output_grad()[0] == Approx(1.0));
}

TEST_CASE("subtract")
{
	nnk::variable<double> x1 = { 1.0 };
	nnk::variable<double> x2 = { 2.0 };

	auto y = x1.expr() - x2.expr();

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(-1.0));

	ev.backward();

	REQUIRE(x1.node()->output_grad().size() == 1);
	CHECK(x1.node()->output_grad()[0] == Approx(1.0));

	REQUIRE(x2.node()->output_grad().size() == 1);
	CHECK(x2.node()->output_grad()[0] == Approx(-1.0));
}

TEST_CASE("multiply")
{
	nnk::variable<double> x1 = { 2.0 };
	nnk::variable<double> x2 = { 3.0 };

	auto y = x1.expr() * x2.expr();

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(6.0));

	ev.backward();

	REQUIRE(x1.node()->output_grad().size() == 1);
	CHECK(x1.node()->output_grad()[0] == Approx(3.0));

	REQUIRE(x2.node()->output_grad().size() == 1);
	CHECK(x2.node()->output_grad()[0] == Approx(2.0));
}

TEST_CASE("divide")
{
	nnk::variable<double> x1 = { 2.0 };
	nnk::variable<double> x2 = { 3.0 };

	auto y = x1.expr() / x2.expr();

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(2.0 / 3.0));

	ev.backward();

	REQUIRE(x1.node()->output_grad().size() == 1);
	CHECK(x1.node()->output_grad()[0] == Approx(1.0 / 3.0));

	REQUIRE(x2.node()->output_grad().size() == 1);
	CHECK(x2.node()->output_grad()[0] == Approx(-2.0 / 9.0));
}
