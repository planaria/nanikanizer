#pragma once
#include <boost/variant.hpp>
#include "piece_type.hpp"

namespace shogi
{

	struct move_action
	{
		int org_row;
		int org_col;
		int new_row;
		int new_col;
		bool promote;
	};

	struct put_action
	{
		piece_type type;
		int row;
		int col;
	};

	typedef boost::variant<move_action, put_action> action;

}
