#pragma once
#include <array>
#include "piece.hpp"
#include "game_state.hpp"
#include "action.hpp"
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
			state_.reset();
			turn_ = initial_turn;
			state_count_.clear();
			++state_count_[state_];
		}

		const game_state& state() const
		{
			return state_;
		}

		bool turn() const
		{
			return turn_;
		}

		bool test_move(int org_row, int org_col, int new_row, int new_col) const
		{
			game_state temp_state = state_;
			return move_impl(temp_state, org_row, org_col, new_row, new_col, false);
		}

		bool move(int org_row, int org_col, int new_row, int new_col, bool promote)
		{
			game_state temp_state = state_;
			bool result = move_impl(temp_state, org_row, org_col, new_row, new_col, promote);

			if (result)
			{
				state_ = temp_state;
				++state_count_[state_];
				turn_ = !turn_;
			}

			return result;
		}

		bool test_put(piece_type type, int row, int col) const
		{
			game_state temp_state = state_;
			return put_impl(temp_state, type, row, col);
		}

		bool put(piece_type type, int row, int col)
		{
			game_state temp_state = state_;
			bool result = put_impl(temp_state, type, row, col);

			if (result)
			{
				state_ = temp_state;
				++state_count_[state_];
				turn_ = !turn_;
			}

			return result;
		}

		bool apply(const action& act)
		{
			if (act.type() == typeid(move_action))
			{
				const move_action& a = boost::get<move_action>(act);
				return move(a.org_row, a.org_col, a.new_row, a.new_col, a.promote);
			}
			else if (act.type() == typeid(put_action))
			{
				const put_action& a = boost::get<put_action>(act);
				return put(a.type, a.row, a.col);
			}

			return false;
		}

	private:

		bool move_impl(game_state& state, int org_row, int org_col, int new_row, int new_col, bool promote) const
		{
			if (org_row < 0 || org_row >= 9 || org_col < 0 || org_col >= 9 ||
				new_row < 0 || new_row >= 9 || new_col < 0 || new_col >= 9)
				return false;

			piece& org_piece = state.table[org_row][org_col];
			piece& new_piece = state.table[new_row][new_col];

			if (org_piece.type() == piece_type::none)
				return false;

			if (org_piece.side() != turn_)
				return false;

			int dx = new_col - org_col;
			int dy = new_row - org_row;
			if (!is_piece_movable(org_piece, dx, dy))
				return false;

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
						return false;

					x += sign_dx;
					y += sign_dy;
				}

				break;
			}
			}

			if (new_piece.type() != piece_type::none)
			{
				if (new_piece.side() == turn_)
					return false;

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
				return false;

			return true;
		}

		bool put_impl(game_state& state, piece_type type, int row, int col) const
		{
			BOOST_ASSERT(type == original(type));

			if (row < 0 || row >= 9 || col < 0 || col >= 9)
				return false;

			hand_type& hand = turn_ ? state.hand2 : state.hand1;
			std::uint8_t& num = hand[static_cast<std::size_t>(type)];

			if (num == 0)
				return false;

			piece& new_piece = state.table[row][col];
			if (new_piece.type() != piece_type::none)
				return false;

			switch (type)
			{
			case piece_type::keima:
				if ((!turn_ && row <= 1) || (turn_ && row >= 7))
					return false;
				break;
			case piece_type::kyosha:
			case piece_type::fuhyo:
				if ((!turn_ && row == 0) || (turn_ && row == 8))
					return false;
				break;
			}

			if (type == piece_type::fuhyo)
			{
				for (std::size_t i = 0; i < 9; ++i)
				{
					const piece& p = state.table[i][col];
					if (p == piece(piece_type::fuhyo, turn_))
						return false;
				}
			}

			new_piece = piece(type, turn_);
			--num;

			if (type == piece_type::fuhyo)
			{
				const piece& forward_piece = state.table[row + (turn_ ? 1 : -1)][col];
				if(forward_piece == piece(piece_type::ousho, !turn_) && is_checkmate(state))
					return false;
			}

			auto count_it = state_count_.find(state);
			if (count_it != state_count_.end() && count_it->second == 3)
				return false;

			return true;
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
