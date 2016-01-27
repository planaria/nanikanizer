#pragma once

namespace nnk
{

	template <class T, class U>
	auto lerp(const T& x1, const T& x2, const U& ratio)
	{
		return x1 + (x2 - x1) * ratio;
	}

	template <class T>
	T clamp(T x, T min, T max)
	{
		if (x < min)
			return min;
		if (max < x)
			return max;
		return x;
	}

	template <class T>
	T square(const T& x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
	{
		return x * x;
	}

	template <class T>
	T norm_sq(const std::valarray<T>& values)
	{
		T result = T();

		for (std::size_t i = 0; i < values.size(); ++i)
			result += square(values[i]);

		return result;
	}

	template <class T>
	T relu(const T& x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
	{
		return std::max(static_cast<T>(0.0), x);
	}

	template <class T>
	T sigmoid(const T& x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
	{
		return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
	}

}
