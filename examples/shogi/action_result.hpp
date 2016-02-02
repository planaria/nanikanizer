#pragma once

namespace shogi
{

	enum class action_result
	{
		succeeded,
		failed,
		win,
	};

	inline std::ostream& operator <<(std::ostream& os, action_result result)
	{
		switch (result)
		{
		case action_result::succeeded:
			os << "succeeded";
			break;
		case action_result::failed:
			os << "failed";
			break;
		case action_result::win:
			os << "win";
			break;
		}

		return os;
	}

}
