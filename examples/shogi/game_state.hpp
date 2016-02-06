#pragma once
#include "piece.hpp"
#include "util.hpp"

namespace shogi
{

	typedef std::array<piece, 9> table_row_type;
	typedef std::array<table_row_type, 9> table_type;
	typedef std::array<std::uint8_t, 15> hand_type;

	struct game_state
	{

		game_state()
		{
			reset();
		}

		void reset()
		{
			for (auto& row : table)
				std::fill(row.begin(), row.end(), piece());

			table[0][0] = piece(piece_type::kyosha, true);
			table[0][1] = piece(piece_type::keima, true);
			table[0][2] = piece(piece_type::ginsho, true);
			table[0][3] = piece(piece_type::kinsho, true);
			table[0][4] = piece(piece_type::ousho, true);
			table[0][5] = piece(piece_type::kinsho, true);
			table[0][6] = piece(piece_type::ginsho, true);
			table[0][7] = piece(piece_type::keima, true);
			table[0][8] = piece(piece_type::kyosha, true);
			table[1][1] = piece(piece_type::hisha, true);
			table[1][7] = piece(piece_type::kaku, true);
			table[2][0] = piece(piece_type::fuhyo, true);
			table[2][1] = piece(piece_type::fuhyo, true);
			table[2][2] = piece(piece_type::fuhyo, true);
			table[2][3] = piece(piece_type::fuhyo, true);
			table[2][4] = piece(piece_type::fuhyo, true);
			table[2][5] = piece(piece_type::fuhyo, true);
			table[2][6] = piece(piece_type::fuhyo, true);
			table[2][7] = piece(piece_type::fuhyo, true);
			table[2][8] = piece(piece_type::fuhyo, true);

			table[6][0] = piece(piece_type::fuhyo, false);
			table[6][1] = piece(piece_type::fuhyo, false);
			table[6][2] = piece(piece_type::fuhyo, false);
			table[6][3] = piece(piece_type::fuhyo, false);
			table[6][4] = piece(piece_type::fuhyo, false);
			table[6][5] = piece(piece_type::fuhyo, false);
			table[6][6] = piece(piece_type::fuhyo, false);
			table[6][7] = piece(piece_type::fuhyo, false);
			table[6][8] = piece(piece_type::fuhyo, false);
			table[7][1] = piece(piece_type::kaku, false);
			table[7][7] = piece(piece_type::hisha, false);
			table[8][0] = piece(piece_type::kyosha, false);
			table[8][1] = piece(piece_type::keima, false);
			table[8][2] = piece(piece_type::ginsho, false);
			table[8][3] = piece(piece_type::kinsho, false);
			table[8][4] = piece(piece_type::ousho, false);
			table[8][5] = piece(piece_type::kinsho, false);
			table[8][6] = piece(piece_type::ginsho, false);
			table[8][7] = piece(piece_type::keima, false);
			table[8][8] = piece(piece_type::kyosha, false);

			std::fill(hand1.begin(), hand1.end(), 0);
			std::fill(hand2.begin(), hand2.end(), 0);
		}

		bool is_win() const
		{
			return hand1[static_cast<std::size_t>(piece_type::ousho)] != 0;
		}

		bool is_lose() const
		{
			return hand2[static_cast<std::size_t>(piece_type::ousho)] != 0;
		}

		bool is_finished() const
		{
			return is_win() || is_lose();
		}

		table_type table;
		hand_type hand1 = {};
		hand_type hand2 = {};

	};

	inline bool operator ==(const game_state& lhs, const game_state& rhs)
	{
		return
			lhs.table == rhs.table &&
			lhs.hand1 == rhs.hand1 &&
			lhs.hand2 == rhs.hand2;
	}

	inline void print_hand(std::ostream& os, const hand_type& hand)
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

	inline std::ostream& operator <<(std::ostream& os, const game_state& state)
	{
		print_hand(os, state.hand2);
		std::cout << "\n";

		for (int i = 0; i < 9; ++i)
		{
			for (int j = 0; j < 9; ++j)
			{
				if (j == 0)
					os << (i == 0 ? "„¡" : "„¥");
				else
					os << (i == 0 ? "„¦" : "„©");
				os << "„Ÿ" << i << j << "„Ÿ";
			}

			os << (i == 0 ? "„¢" : "„§");
			os << "\n";

			os << "„ ";

			for (auto& p : state.table[i])
				os << p << "„ ";

			os << "\n";
		}

		os << "„¤„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„¨„Ÿ„Ÿ„Ÿ„£\n";

		print_hand(os, state.hand1);
		std::cout << "\n";

		return os;
	}

}

namespace std
{

	template <>
	struct hash<shogi::game_state>
	{

		std::size_t operator ()(const shogi::game_state& state) const
		{
			std::size_t hash = 0;
			shogi::hash_combine(hash, state.table);
			shogi::hash_combine(hash, state.hand1);
			shogi::hash_combine(hash, state.hand2);
			return hash;
		}

	};

	template <>
	struct hash<shogi::table_type>
	{

		std::size_t operator ()(const shogi::table_type& table) const
		{
			std::size_t hash = 0;
			for (const auto& row : table)
				shogi::hash_combine(hash, row);
			return hash;
		}

	};

	template <>
	struct hash<shogi::table_row_type>
	{

		std::size_t operator ()(const shogi::table_row_type& row) const
		{
			std::size_t hash = 0;
			for (shogi::piece p : row)
				shogi::hash_combine(hash, p);
			return hash;
		}

	};

	template <>
	struct hash<shogi::hand_type>
	{

		std::size_t operator ()(const shogi::hand_type& hand) const
		{
			std::size_t hash = 0;
			for (std::uint8_t num : hand)
				shogi::hash_combine(hash, num);
			return hash;
		}

	};

}
