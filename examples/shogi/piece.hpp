#pragma once
#include "piece_type.hpp"
#include "util.hpp"

namespace shogi
{

	class piece
		: boost::equality_comparable<piece>
	{
	public:

		piece()
		{
		}

		piece(piece_type type, bool side)
		{
			BOOST_ASSERT(type >= piece_type::none && type < piece_type::end);
			data_ = side ? -static_cast<std::int8_t>(type) : static_cast<std::int8_t>(type);
		}

		std::int8_t data() const
		{
			return data_;
		}

		piece_type type() const
		{
			return static_cast<piece_type>(std::abs(data_));
		}

		bool side() const
		{
			BOOST_ASSERT(type() != piece_type::none);
			return data_ < 0;
		}

	private:

		std::int8_t data_ = 0;

	};

	inline bool operator ==(piece lhs, piece rhs)
	{
		return lhs.data() == rhs.data();
	}

	inline bool is_piece_movable(piece p, int dx, int dy)
	{
		if (dx == 0 && dy == 0)
			return false;

		if (!p.side())
			dy = -dy;

		dx = std::abs(dx);

		switch (p.type())
		{
		case piece_type::ousho:
			if (dx <= 1 && std::abs(dy) <= 1)
				return true;
			break;
		case piece_type::kinsho:
		case piece_type::narigin:
		case piece_type::narikei:
		case piece_type::narikyo:
		case piece_type::tokin:
			if (dx <= 1 && (dy == 1 || dy == 0))
				return true;
			if (dx == 0 && dy == -1)
				return true;
			break;
		case piece_type::ginsho:
			if (dx <= 1 && dy == 1)
				return true;
			if (dx == 1 && dy == -1)
				return true;
			break;
		case piece_type::keima:
			if (dx == 1 && dy == 2)
				return true;
			break;
		case piece_type::kyosha:
			if (dx == 0 && dy >= 1)
				return true;
		case piece_type::kaku:
			if (dx == dy || dx == -dy)
				return true;
			break;
		case piece_type::ryuma:
			if (dx == dy || dx == -dy)
				return true;
			if (dx <= 1 && std::abs(dy) <= 1)
				return true;
			break;
		case piece_type::hisha:
			if (dx == 0 || dy == 0)
				return true;
			break;
		case piece_type::ryuou:
			if (dx == 0 || dy == 0)
				return true;
			if (dx <= 1 && std::abs(dy) <= 1)
				return true;
			break;
		case piece_type::fuhyo:
			if (dx == 0 && dy == 1)
				return true;
			break;
		}

		return false;
	}

	inline std::ostream& operator <<(std::ostream& os, piece p)
	{
		if (p.type() == piece_type::none)
		{
			os << "      ";
		}
		else
		{
			os << (p.side() ? "«" : "ª");

			static const char* name[] =
			{
				"    ",
				"‰¤  ",
				"‹à  ",
				"‹â  ",
				"¬‹â",
				"Œj  ",
				"¬Œj",
				"  ",
				"¬",
				"Šp  ",
				"”n  ",
				"”ò  ",
				"—´  ",
				"•à  ",
				"‚Æ  ",
			};

			if (p.side() && p.type() == piece_type::ousho)
				os << "‹Ê  ";
			else
				os << name[static_cast<std::int8_t>(p.type())];
		}

		return os;
	}

}


namespace std
{

	template <>
	struct hash<shogi::piece>
	{

		std::size_t operator ()(const shogi::piece& p) const
		{
			std::size_t hash = 0;
			shogi::hash_combine(hash, p.data());
			return hash;
		}

	};

}
