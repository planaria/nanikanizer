#pragma once

namespace nnk
{

	template <class T>
	std::vector<std::valarray<T>> make_ids(std::size_t size, T false_value, T true_value)
	{
		std::vector<std::valarray<T>> result;

		for (std::size_t i = 0; i < size; ++i)
		{
			std::valarray<T> id(false_value, size);
			id[i] = true_value;

			result.push_back(std::move(id));
		}

		return result;
	}

}
