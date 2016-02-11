#pragma once
#include <cstdint>

namespace shogi
{

	enum class piece_type : std::int8_t
	{
		none,
		ousho,
		kinsho,
		ginsho,
		narigin,
		keima,
		narikei,
		kyosha,
		narikyo,
		kaku,
		ryuma,
		hisha,
		ryuou,
		fuhyo,
		tokin,
		end,
	};

	inline piece_type original(piece_type type)
	{
		switch (type)
		{
		case piece_type::narigin:
			return piece_type::ginsho;
		case piece_type::narikei:
			return piece_type::keima;
		case piece_type::narikyo:
			return piece_type::kyosha;
		case piece_type::ryuma:
			return piece_type::kaku;
		case piece_type::ryuou:
			return piece_type::hisha;
		case piece_type::tokin:
			return piece_type::fuhyo;
		}

		return type;
	}

	inline piece_type promoted(piece_type type)
	{
		switch (type)
		{
		case piece_type::ginsho:
			return piece_type::narigin;
		case piece_type::keima:
			return piece_type::narikei;
		case piece_type::kyosha:
			return piece_type::narikyo;
		case piece_type::kaku:
			return piece_type::ryuma;
		case piece_type::hisha:
			return piece_type::ryuou;
		case piece_type::fuhyo:
			return piece_type::tokin;
		}

		return type;
	}

	inline std::ostream& operator <<(std::ostream& os, piece_type type)
	{
		static const char* name[] =
		{
			"",
			"‰¤",
			"‹à",
			"‹â",
			"¬‹â",
			"Œj",
			"¬Œj",
			"",
			"¬",
			"Šp",
			"”n",
			"”ò",
			"—´",
			"•à",
			"‚Æ",
		};

		os << name[static_cast<std::int8_t>(type)];

		return os;
	}

}
