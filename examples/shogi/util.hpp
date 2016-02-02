#pragma once

namespace shogi
{

	template <class T>
	int sign(T x)
	{
		if (x < 0)
			return -1;
		if (x > 0)
			return 1;
		return 0;
	}

	template <class T>
	void hash_combine(std::size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

}
