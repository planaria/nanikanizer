#include <iostream>
#include <iomanip>
#include <nanikanizer/nanikanizer.hpp>
#include "game.hpp"

namespace shogi
{

	static const std::size_t num_ousho = 2;
	static const std::size_t num_kinsho = 4;
	static const std::size_t num_ginsho = 4;
	static const std::size_t num_keima = 4;
	static const std::size_t num_kyosha = 4;
	static const std::size_t num_kaku = 2;
	static const std::size_t num_hisha = 2;
	static const std::size_t num_fusho = 18;

	static const std::size_t num_piece =
		num_ousho + num_kinsho + num_ginsho + num_keima +
		num_kyosha + num_kaku + num_hisha + num_fusho;

	static const std::size_t input_size = 9 * 9 * 29 + num_piece * 2;

	void apply(float* data, const hand_type& hand)
	{
		std::size_t data_index = 0;

		for (std::size_t i = 0; i < num_ousho; ++i)
			data[data_index++] = hand[static_cast<std::size_t>(piece_type::ousho)] > i ? 1.0f : 0.0f;

		for (std::size_t i = 0; i < num_kinsho; ++i)
			data[data_index++] = hand[static_cast<std::size_t>(piece_type::kinsho)] > i ? 1.0f : 0.0f;

		for (std::size_t i = 0; i < num_ginsho; ++i)
			data[data_index++] = hand[static_cast<std::size_t>(piece_type::ginsho)] > i ? 1.0f : 0.0f;

		for (std::size_t i = 0; i < num_keima; ++i)
			data[data_index++] = hand[static_cast<std::size_t>(piece_type::keima)] > i ? 1.0f : 0.0f;

		for (std::size_t i = 0; i < num_kyosha; ++i)
			data[data_index++] = hand[static_cast<std::size_t>(piece_type::kyosha)] > i ? 1.0f : 0.0f;

		for (std::size_t i = 0; i < num_kaku; ++i)
			data[data_index++] = hand[static_cast<std::size_t>(piece_type::kaku)] > i ? 1.0f : 0.0f;

		for (std::size_t i = 0; i < num_hisha; ++i)
			data[data_index++] = hand[static_cast<std::size_t>(piece_type::hisha)] > i ? 1.0f : 0.0f;

		for (std::size_t i = 0; i < num_fusho; ++i)
			data[data_index++] = hand[static_cast<std::size_t>(piece_type::fuhyo)] > i ? 1.0f : 0.0f;

		BOOST_ASSERT(data_index == num_piece);
	}

	void apply(float* data, const game_state& state)
	{
		std::fill_n(data, input_size, 0.0f);

		std::size_t data_index = 0;

		for (std::size_t row = 0; row < 9; ++row)
		{
			for (std::size_t col = 0; col < 9; ++col)
			{
				const piece& p = state.table[row][col];
				std::size_t index = static_cast<std::size_t>(p.type());
				if (p != piece() && p.side())
					index += 14;

				data[data_index + index] = 1.0f;

				data_index += 29;
			}
		}

		apply(data + data_index, state.hand1);
		data_index += num_piece;

		apply(data + data_index, state.hand2);
		data_index += num_piece;

		BOOST_ASSERT(data_index == input_size);
	}

}

int main(int /*argc*/, char* /*argv*/[])
{
	using namespace shogi;

	try
	{
		game g;

		nnk::variable<float> input(input_size);

		nnk::bidirectional_linear_layer<float> l1(input_size, 1000);

		nnk::expression<float> x = input.expr();

		apply(&input.value()[0], g.state());
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
