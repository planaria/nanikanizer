#pragma once
#include "game.hpp"

namespace shogi
{

	template <class Iterator>
	void enumerate_next(const game& g, Iterator it)
	{
		for (int row = 0; row < 9; ++row)
		{
			for (int col = 0; col < 9; ++col)
			{
				const piece& p = g.state().table[row][col];

				if (p.type() == piece_type::none || p.side() != g.turn())
					continue;

				auto try_move = [&](int new_row, int new_col)
				{
					if (g.test_move(row, col, new_row, new_col) == action_result::failed)
						return false;

					game next = g;
					next.move(row, col, new_row, new_col, true);

					*it++ = std::move(next);
					return true;
				};

				int dy = g.turn() ? 1 : -1;

				switch (p.type())
				{
				case piece_type::ousho:
					try_move(row - 1, col - 1);
					try_move(row - 1, col);
					try_move(row - 1, col + 1);
					try_move(row, col - 1);
					try_move(row, col + 1);
					try_move(row + 1, col - 1);
					try_move(row + 1, col);
					try_move(row + 1, col + 1);
					break;
				case piece_type::kinsho:
				case piece_type::narigin:
				case piece_type::narikei:
				case piece_type::narikyo:
				case piece_type::tokin:
					try_move(row + dy, col - 1);
					try_move(row + dy, col);
					try_move(row + dy, col + 1);
					try_move(row, col - 1);
					try_move(row, col + 1);
					try_move(row - dy, col);
					break;
				case piece_type::ginsho:
					try_move(row + dy, col - 1);
					try_move(row + dy, col);
					try_move(row + dy, col + 1);
					try_move(row - dy, col - 1);
					try_move(row - dy, col + 1);
					break;
				case piece_type::keima:
					try_move(row + dy * 2, col - 1);
					try_move(row + dy * 2, col + 1);
					break;
				case piece_type::kyosha:
					for (int i = 1; i < 9; ++i)
						if (!try_move(row + dy * i, col))
							break;
					break;
				case piece_type::kaku:
					for (int i = 1; i < 9; ++i)
						if (!try_move(row + i, col + i))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row + i, col - i))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row - i, col + i))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row - i, col - i))
							break;
					break;
				case piece_type::ryuma:
					for (int i = 1; i < 9; ++i)
						if (!try_move(row + i, col + i))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row + i, col - i))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row - i, col + i))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row - i, col - i))
							break;
					try_move(row - 1, col);
					try_move(row, col - 1);
					try_move(row, col + 1);
					try_move(row + 1, col);
					break;
				case piece_type::hisha:
					for (int i = 1; i < 9; ++i)
						if (!try_move(row, col + i))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row, col - i))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row + i, col))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row - i, col))
							break;
					break;
				case piece_type::ryuou:
					for (int i = 1; i < 9; ++i)
						if (!try_move(row, col + i))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row, col - i))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row + i, col))
							break;
					for (int i = 1; i < 9; ++i)
						if (!try_move(row - i, col))
							break;
					try_move(row - 1, col - 1);
					try_move(row - 1, col + 1);
					try_move(row + 1, col - 1);
					try_move(row + 1, col + 1);
					break;
				case piece_type::fuhyo:
					try_move(row + dy, col);
					break;
				}
			}
		}

		piece_type hand_types[] =
		{
			piece_type::kinsho,
			piece_type::ginsho,
			piece_type::keima,
			piece_type::kyosha,
			piece_type::kaku,
			piece_type::hisha,
			piece_type::fuhyo,
		};

		for (piece_type type : hand_types)
		{
			const hand_type& hand = g.turn() ? g.state().hand2 : g.state().hand1;
			if (hand[static_cast<std::size_t>(type)] == 0)
				continue;

			for (int row = 0; row < 9; ++row)
			{
				for (int col = 0; col < 9; ++col)
				{
					const piece& p = g.state().table[row][col];

					if (p.type() != piece_type::none)
						continue;

					if (g.test_put(type, row, col) == action_result::failed)
						continue;

					game next = g;
					next.put(type, row, col);

					*it++ = std::move(next);
				}
			}
		}
	}

}
