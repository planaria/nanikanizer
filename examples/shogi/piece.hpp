#pragma once
#include "piece_type.hpp"

namespace shogi
{

	class piece
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

		piece_type type() const
		{
			return static_cast<piece_type>(std::abs(data_));
		}

		bool side() const
		{
			return data_ < 0;
		}

	private:

		std::int8_t data_ = 0;

	};

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
