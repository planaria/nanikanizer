#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("matrix_product")
{
	nnk::variable<double> x1 =
	{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	};

	nnk::variable<double> x2 =
	{
		1.0, 2.0,
		3.0, 4.0,
		5.0, 6.0,
	};

	auto y = matrix_product(x1.expr(), x2.expr(), 2, 3, 3, 2);

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 4);
	CHECK(result[0] == Approx(22.0));
	CHECK(result[1] == Approx(28.0));
	CHECK(result[2] == Approx(49.0));
	CHECK(result[3] == Approx(64.0));

	ev.backward({ 1.0, 1.0, 1.0, 1.0 });

	REQUIRE(x1.node()->output_grad().size() == 6);
	CHECK(x1.node()->output_grad()[0] == Approx(3.0));
	CHECK(x1.node()->output_grad()[1] == Approx(7.0));
	CHECK(x1.node()->output_grad()[2] == Approx(11.0));
	CHECK(x1.node()->output_grad()[3] == Approx(3.0));
	CHECK(x1.node()->output_grad()[4] == Approx(7.0));
	CHECK(x1.node()->output_grad()[5] == Approx(11.0));

	REQUIRE(x2.node()->output_grad().size() == 6);
	CHECK(x2.node()->output_grad()[0] == Approx(5.0));
	CHECK(x2.node()->output_grad()[1] == Approx(5.0));
	CHECK(x2.node()->output_grad()[2] == Approx(7.0));
	CHECK(x2.node()->output_grad()[3] == Approx(7.0));
	CHECK(x2.node()->output_grad()[4] == Approx(9.0));
	CHECK(x2.node()->output_grad()[5] == Approx(9.0));
}

TEST_CASE("matrix_transpose")
{
	nnk::variable<double> x =
	{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	};

	auto y = matrix_transpose(x.expr(), 2, 3);

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 6);
	CHECK(result[0] == Approx(1.0));
	CHECK(result[1] == Approx(4.0));
	CHECK(result[2] == Approx(2.0));
	CHECK(result[3] == Approx(5.0));
	CHECK(result[4] == Approx(3.0));
	CHECK(result[5] == Approx(6.0));

	ev.backward({ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

	REQUIRE(x.node()->output_grad().size() == 6);
	CHECK(x.node()->output_grad()[0] == Approx(1.0));
	CHECK(x.node()->output_grad()[1] == Approx(1.0));
	CHECK(x.node()->output_grad()[2] == Approx(1.0));
	CHECK(x.node()->output_grad()[3] == Approx(1.0));
	CHECK(x.node()->output_grad()[4] == Approx(1.0));
	CHECK(x.node()->output_grad()[5] == Approx(1.0));
}
