#pragma once

namespace nnk
{

	template <class T, class U>
	auto lerp(const T& x1, const T& x2, const U& ratio)
	{
		return x1 + (x2 - x1) * ratio;
	}

	template <class T>
	T pow2(const T& x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
	{
		return x * x;
	}

	template <class T>
	T norm_sq(const std::valarray<T>& values)
	{
		T result = T();

		for (std::size_t i = 0; i < values.size(); ++i)
			result += pow2(values[i]);

		return result;
	}

	template <class T>
	T sign(const T& x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
	{
		return x < 0 ? static_cast<T>(-1) : static_cast<T>(1);
	}

	template <class T>
	T dot(const std::valarray<T>& lhs, const std::valarray<T>& rhs)
	{
		BOOST_ASSERT(lhs.size() == rhs.size());

		T result = T();

		for (std::size_t i = 0; i < lhs.size(); ++i)
			result += lhs[i] * rhs[i];

		return result;
	}

	template <class T>
	T relu(const T& x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
	{
		return std::max(static_cast<T>(0.0), x);
	}

	template <class T>
	T relu_diff(const T& x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
	{
		return x < static_cast<T>(0.0) ? static_cast<T>(0.0) : static_cast<T>(1.0);
	}

	template <class T>
	T sigmoid(const T& x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
	{
		return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
	}

	template <class T>
	T sigmoid_diff(const T& x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
	{
		return x * (static_cast<T>(1.0) - x);
	}

	template <class T>
	T tanh_diff(const T& x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
	{
		return static_cast<T>(1.0) - x * x;
	}

}
