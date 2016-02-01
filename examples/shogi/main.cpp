#include <iostream>
#include <iomanip>
#include <nanikanizer/nanikanizer.hpp>
#include "game.hpp"

int main(int /*argc*/, char* /*argv*/[])
{
	try
	{
		shogi::game g;

		std::cout << g << std::endl;
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
