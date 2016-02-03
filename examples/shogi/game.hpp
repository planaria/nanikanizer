#pragma once
#include <array>
#include "piece.hpp"
#include "game_state.hpp"
#include "action_result.hpp"
#include "util.hpp"

namespace shogi
{

	class game
	{
	public:

		explicit game(bool initial_turn = false)
		{
			reset(initial_turn);
		}

		void reset(bool initial_turn = false)
		{
			turn_ = initial_turn;

			state_.table[0][0] = piece(piece_type::kyosha, true);
			state_.table[0][1] = piece(piece_type::keima, true);
			state_.table[0][2] = piece(piece_type::ginsho, true);
			state_.table[0][3] = piece(piece_type::kinsho, true);
			state_.table[0][4] = piece(piece_type::ousho, true);
			state_.table[0][5] = piece(piece_type::kinsho, true);
			state_.table[0][6] = piece(piece_type::ginsho, true);
			state_.table[0][7] = piece(piece_type::keima, true);
			state_.table[0][8] = piece(piece_type::kyosha, true);
			state_.table[1][1] = piece(piece_type::hisha, true);
			state_.table[1][7] = piece(piece_type::kaku, true);
			state_.table[2][0] = piece(piece_type::fuhyo, true);
			state_.table[2][1] = piece(piece_type::fuhyo, true);
			state_.table[2][2] = piece(piece_type::fuhyo, true);
			state_.table[2][3] = piece(piece_type::fuhyo, true);
			state_.table[2][4] = piece(piece_type::fuhyo, true);
			state_.table[2][5] = piece(piece_type::fuhyo, true);
			state_.table[2][6] = piece(piece_type::fuhyo, true);
			state_.table[2][7] = piece(piece_type::fuhyo, true);
			state_.table[2][8] = piece(piece_type::fuhyo, true);

			state_.table[6][0] = piece(piece_type::fuhyo, false);
			state_.table[6][1] = piece(piece_type::fuhyo, false);
			state_.table[6][2] = piece(piece_type::fuhyo, false);
			state_.table[6][3] = piece(piece_type::fuhyo, false);
			state_.table[6][4] = piece(piece_type::fuhyo, false);
			state_.table[6][5] = piece(piece_type::fuhyo, false);
			state_.table[6][6] = piece(piece_type::fuhyo, false);
			state_.table[6][7] = piece(piece_type::fuhyo, false);
			state_.table[6][8] = piece(piece_type::fuhyo, false);
			state_.table[7][1] = piece(piece_type::kaku, false);
			state_.table[7][7] = piece(piece_type::hisha, false);
			state_.table[8][0] = piece(piece_type::kyosha, false);
			state_.table[8][1] = piece(piece_type::keima, false);
			state_.table[8][2] = piece(piece_type::ginsho, false);
			state_.table[8][3] = piece(piece_type::kinsho, false);
			state_.table[8][4] = piece(piece_type::ousho, false);
			state_.table[8][5] = piece(piece_type::kinsho, false);
			state_.table[8][6] = piece(piece_type::ginsho, false);
			state_.table[8][7] = piece(piece_type::keima, false);
			state_.table[8][8] = piece(piece_type::kyosha, false);

			state_count_.clear();

			++state_count_[state_];
		}

		const game_state& state() const
		{
			return state_;
		}

		action_result test_move(int org_row, int org_col, int new_row, int new_col) const
		{
			game_state temp_state = state_;
			return move_impl(temp_state, org_row, org_col, new_row, new_col, false);
		}

		action_result move(int org_row, int org_col, int new_row, int new_col, bool promote)
		{
			game_state temp_state = state_;
			action_result result = move_impl(temp_state, org_row, org_col, new_row, new_col, promote);

			if (result != action_result::failed)
			{
				state_ = temp_state;
				++state_count_[state_];
				turn_ = !turn_;
			}

			return result;
		}

		action_result test_put(piece_type type, int row, int col) const
		{
			game_state temp_state = state_;
			return put_impl(temp_state, type, row, col);
		}

		action_result put(piece_type type, int row, int col)
		{
			game_state temp_state = state_;
			action_result result = put_impl(temp_state, type, row, col);

			if (result != action_result::failed)
			{
				state_ = temp_state;
				++state_count_[state_];
				turn_ = !turn_;
			}

			return result;
		}

	private:

		action_result move_impl(game_state& state, int org_row, int org_col, int new_row, int new_col, bool promote) const
		{
			BOOST_ASSERT(org_row >= 0 && org_row < 9);
			BOOST_ASSERT(org_col >= 0 && org_col < 9);
			BOOST_ASSERT(new_row >= 0 && new_row < 9);
			BOOST_ASSERT(new_col >= 0 && new_col < 9);

			piece& org_piece = state.table[org_row][org_col];
			piece& new_piece = state.table[new_row][new_col];

			if (org_piece.type() == piece_type::none)
				return action_result::failed;

			if (org_piece.side() != turn_)
				return action_result::failed;

			int dx = new_col - org_col;
			int dy = new_row - org_row;
			if (!is_piece_movable(org_piece, dx, dy))
				return action_result::failed;

			switch (org_piece.type())
			{
			case piece_type::kaku:
			case piece_type::ryuma:
			case piece_type::hisha:
			case piece_type::ryuou:
			case piece_type::kyosha:
			{
				int sign_dx = sign(dx);
				int sign_dy = sign(dy);
				int x = org_col + sign_dx;
				int y = org_row + sign_dy;

				while (x != new_col || y != new_row)
				{
					const piece& p = state.table[y][x];

					if (p.type() != piece_type::none)
						return action_result::failed;

					x += sign_dx;
					y += sign_dy;
				}

				break;
			}
			}

			piece_type got_piece_type = piece_type::none;

			if (new_piece.type() != piece_type::none)
			{
				if (new_piece.side() == turn_)
					return action_result::failed;

				got_piece_type = new_piece.type();

				hand_type& hand = turn_ ? state.hand2 : state.hand1;
				++hand[static_cast<std::size_t>(original(new_piece.type()))];
			}

			new_piece = org_piece;
			org_piece = piece();

			if (!promote)
			{
				switch (new_piece.type())
				{
				case piece_type::keima:
					if ((!turn_ && new_row <= 1) || (turn_ && new_row >= 7))
						promote = true;
					break;
				case piece_type::kyosha:
				case piece_type::fuhyo:
					if ((!turn_ && new_row == 0) || (turn_ && new_row == 8))
						promote = true;
					break;
				}
			}

			if (promote && ((!turn_ && new_row <= 2) || (turn_ && new_row >= 6)))
				new_piece = piece(promoted(new_piece.type()), new_piece.side());

			auto count_it = state_count_.find(state);
			if (count_it != state_count_.end() && count_it->second == 3)
				return action_result::failed;

			if (got_piece_type == piece_type::ousho)
				return action_result::win;

			return action_result::succeeded;
		}

		action_result put_impl(game_state& state, piece_type type, int row, int col) const
		{
			BOOST_ASSERT(type == original(type));
			BOOST_ASSERT(row >= 0 && row < 9);
			BOOST_ASSERT(col >= 0 && col < 9);

			hand_type& hand = turn_ ? state.hand2 : state.hand1;
			std::uint8_t& num = hand[static_cast<std::size_t>(type)];

			if (num == 0)
				return action_result::failed;

			piece& new_piece = state.table[row][col];
			if (new_piece.type() != piece_type::none)
				return action_result::failed;

			switch (type)
			{
			case piece_type::keima:
				if ((!turn_ && row <= 1) || (turn_ && row >= 7))
					return action_result::failed;
				break;
			case piece_type::kyosha:
			case piece_type::fuhyo:
				if ((!turn_ && row == 0) || (turn_ && row == 8))
					return action_result::failed;
				break;
			}

			if (type == piece_type::fuhyo)
			{
				for (std::size_t i = 0; i < 9; ++i)
				{
					const piece& p = state.table[i][col];
					if (p == piece(piece_type::fuhyo, turn_))
						return action_result::failed;
				}
			}

			new_piece = piece(type, turn_);
			--num;

			if (type == piece_type::fuhyo)
			{
				const piece& forward_piece = state.table[row + (turn_ ? 1 : -1)][col];
				if(forward_piece == piece(piece_type::ousho, !turn_) && is_checkmate(state))
					return action_result::failed;
			}

			auto count_it = state_count_.find(state);
			if (count_it != state_count_.end() && count_it->second == 3)
				return action_result::failed;

			return action_result::succeeded;
		}

		bool is_checkmate(const game_state& state) const
		{
			// TODO
			return true;
		}

		game_state state_;
		bool turn_;

		std::unordered_map<game_state, std::uint8_t> state_count_;

	};

}
