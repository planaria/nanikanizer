#pragma once
#include <array>
#include "piece.hpp"

namespace shogi
{

	class game
	{
	public:

		typedef std::array<piece, 9> row_type;
		typedef std::array<row_type, 9> table_type;
		typedef std::array<std::uint8_t, 15> hand_type;

		game()
		{
			reset();
		}

		void reset()
		{
			for (auto& row : table_)
				std::fill(row.begin(), row.end(), piece());

			table_[0][0] = piece(piece_type::kyosha, true);
			table_[0][1] = piece(piece_type::keima, true);
			table_[0][2] = piece(piece_type::ginsho, true);
			table_[0][3] = piece(piece_type::kinsho, true);
			table_[0][4] = piece(piece_type::ousho, true);
			table_[0][5] = piece(piece_type::kinsho, true);
			table_[0][6] = piece(piece_type::ginsho, true);
			table_[0][7] = piece(piece_type::keima, true);
			table_[0][8] = piece(piece_type::kyosha, true);
			table_[1][1] = piece(piece_type::hisha, true);
			table_[1][7] = piece(piece_type::kaku, true);
			table_[2][0] = piece(piece_type::fuhyo, true);
			table_[2][1] = piece(piece_type::fuhyo, true);
			table_[2][2] = piece(piece_type::fuhyo, true);
			table_[2][3] = piece(piece_type::fuhyo, true);
			table_[2][4] = piece(piece_type::fuhyo, true);
			table_[2][5] = piece(piece_type::fuhyo, true);
			table_[2][6] = piece(piece_type::fuhyo, true);
			table_[2][7] = piece(piece_type::fuhyo, true);
			table_[2][8] = piece(piece_type::fuhyo, true);

			table_[6][0] = piece(piece_type::fuhyo, false);
			table_[6][1] = piece(piece_type::fuhyo, false);
			table_[6][2] = piece(piece_type::fuhyo, false);
			table_[6][3] = piece(piece_type::fuhyo, false);
			table_[6][4] = piece(piece_type::fuhyo, false);
			table_[6][5] = piece(piece_type::fuhyo, false);
			table_[6][6] = piece(piece_type::fuhyo, false);
			table_[6][7] = piece(piece_type::fuhyo, false);
			table_[6][8] = piece(piece_type::fuhyo, false);
			table_[7][1] = piece(piece_type::kaku, false);
			table_[7][7] = piece(piece_type::hisha, false);
			table_[8][0] = piece(piece_type::kyosha, false);
			table_[8][1] = piece(piece_type::keima, false);
			table_[8][2] = piece(piece_type::ginsho, false);
			table_[8][3] = piece(piece_type::kinsho, false);
			table_[8][4] = piece(piece_type::ousho, false);
			table_[8][5] = piece(piece_type::kinsho, false);
			table_[8][6] = piece(piece_type::ginsho, false);
			table_[8][7] = piece(piece_type::keima, false);
			table_[8][8] = piece(piece_type::kyosha, false);
		}

		const table_type& table() const
		{
			return table_;
		}

		const hand_type& hand1() const
		{
			return hand1_;
		}

		const hand_type& hand2() const
		{
			return hand2_;
		}

		bool move(int org_row, int org_col, int new_row, int new_org, bool promote)
		{
			BOOST_ASSERT(org_row >= 0 && org_row < 9);
			BOOST_ASSERT(org_col >= 0 && org_col < 9);
			BOOST_ASSERT(new_row >= 0 && new_row < 9);
			BOOST_ASSERT(new_org >= 0 && new_org < 9);

			return false;
		}

		bool put(piece_type type, int row, int col)
		{
			BOOST_ASSERT(type == original(type));
			BOOST_ASSERT(row >= 0 && row < 9);
			BOOST_ASSERT(col >= 0 && col < 9);

			return false;
		}

	private:

		table_type table_;

		hand_type hand1_ = {};
		hand_type hand2_ = {};

	};

	inline void print_hand(std::ostream& os, const game::hand_type& hand)
	{
		bool first = true;

		for (std::size_t i = 0; i < static_cast<std::size_t>(piece_type::end); ++i)
		{
			std::uint32_t num = hand[i];
			if (num != 0)
			{
				if (!first)
					os << " ";

				os << static_cast<piece_type>(i) << num;

				first = false;
			}
		}
	}

	inline std::ostream& operator <<(std::ostream& os, game& g)
	{
		print_hand(os, g.hand1());
		std::cout << "\n";

		os << "„¡„Ÿ„Ÿ„Ÿ„¦„Ÿ„Ÿ„Ÿ„¦„Ÿ„Ÿ„Ÿ„¦„Ÿ„Ÿ„Ÿ„¦„Ÿ„Ÿ„Ÿ„¦„Ÿ„Ÿ„Ÿ„¦„Ÿ„Ÿ„Ÿ„¦„Ÿ„Ÿ„Ÿ„¦„Ÿ„Ÿ„Ÿ„¢\n";

		for (int i = 0; i < 9; ++i)
		{
			if(i != 0)
				os << "„¥„Ÿ„Ÿ„Ÿ„©„Ÿ„Ÿ„Ÿ„©„Ÿ„Ÿ„Ÿ„©„Ÿ„Ÿ„Ÿ„©„Ÿ„Ÿ„Ÿ„©„Ÿ„Ÿ„Ÿ„©„Ÿ„Ÿ„Ÿ„©„Ÿ„Ÿ„Ÿ„©„Ÿ„Ÿ„Ÿ„§\n";

			os << "„ ";

			for (auto& p : g.table()[i])
				os << p << "„ ";

			os << "\n";
		}

		os << "„¤„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„£\n";

		print_hand(os, g.hand2());
		std::cout << "\n";

		return os;
	}

}
