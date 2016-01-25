#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("sum")
{
	nnk::variable<double> x = { 2.0, 3.0 };

	auto y = sum(x.expr());

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(5.0));

	ev.backward();

	REQUIRE(x.node()->output_grad().size() == 2);
	CHECK(x.node()->output_grad()[0] == Approx(1.0));
	CHECK(x.node()->output_grad()[1] == Approx(1.0));
}

TEST_CASE("norm")
{
	nnk::variable<double> x = { 2.0, 3.0 };

	auto y = norm(x.expr());

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(std::sqrt(13.0)));

	ev.backward();

	REQUIRE(x.node()->output_grad().size() == 2);
	CHECK(x.node()->output_grad()[0] == Approx(2.0 / std::sqrt(13.0)));
	CHECK(x.node()->output_grad()[1] == Approx(3.0 / std::sqrt(13.0)));
}

TEST_CASE("norm_sq")
{
	nnk::variable<double> x = { 2.0, 3.0 };

	auto y = norm_sq(x.expr());

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 1);
	CHECK(result[0] == Approx(13.0));

	ev.backward();

	REQUIRE(x.node()->output_grad().size() == 2);
	CHECK(x.node()->output_grad()[0] == Approx(4.0));
	CHECK(x.node()->output_grad()[1] == Approx(6.0));
}