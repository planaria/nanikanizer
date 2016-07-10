#include <catch.hpp>
#include <nanikanizer/nanikanizer.hpp>

TEST_CASE("skipping_2d")
{
	nnk::variable<double> x =
	{
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
		7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
		13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
		19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
		25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
	};

	auto y = nnk::skipping_2d(x.expr(), 5, 6, 1, 1, 1);

	nnk::evaluator<double> ev(y);

	auto result = ev.forward();

	REQUIRE(result.size() == 9);
	CHECK(result[0] == Approx(1.0));
	CHECK(result[1] == Approx(3.0));
	CHECK(result[2] == Approx(5.0));
	CHECK(result[3] == Approx(13.0));
	CHECK(result[4] == Approx(15.0));
	CHECK(result[5] == Approx(17.0));
	CHECK(result[6] == Approx(25.0));
	CHECK(result[7] == Approx(27.0));
	CHECK(result[8] == Approx(29.0));

	ev.backward(
	{
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
	});

	REQUIRE(x.grad().size() == 30);
	CHECK(x.grad()[0] == Approx(1.0));
	CHECK(x.grad()[1] == Approx(0.0));
	CHECK(x.grad()[2] == Approx(1.0));
	CHECK(x.grad()[3] == Approx(0.0));
	CHECK(x.grad()[4] == Approx(1.0));
	CHECK(x.grad()[5] == Approx(0.0));
	CHECK(x.grad()[6] == Approx(0.0));
	CHECK(x.grad()[7] == Approx(0.0));
	CHECK(x.grad()[8] == Approx(0.0));
	CHECK(x.grad()[9] == Approx(0.0));
	CHECK(x.grad()[10] == Approx(0.0));
	CHECK(x.grad()[11] == Approx(0.0));
	CHECK(x.grad()[12] == Approx(1.0));
	CHECK(x.grad()[13] == Approx(0.0));
	CHECK(x.grad()[14] == Approx(1.0));
	CHECK(x.grad()[15] == Approx(0.0));
	CHECK(x.grad()[16] == Approx(1.0));
	CHECK(x.grad()[17] == Approx(0.0));
	CHECK(x.grad()[18] == Approx(0.0));
	CHECK(x.grad()[19] == Approx(0.0));
	CHECK(x.grad()[20] == Approx(0.0));
	CHECK(x.grad()[21] == Approx(0.0));
	CHECK(x.grad()[22] == Approx(0.0));
	CHECK(x.grad()[23] == Approx(0.0));
	CHECK(x.grad()[24] == Approx(1.0));
	CHECK(x.grad()[25] == Approx(0.0));
	CHECK(x.grad()[26] == Approx(1.0));
	CHECK(x.grad()[27] == Approx(0.0));
	CHECK(x.grad()[28] == Approx(1.0));
	CHECK(x.grad()[29] == Approx(0.0));
}
