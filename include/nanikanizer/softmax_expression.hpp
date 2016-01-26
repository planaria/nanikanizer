#pragma once
#include "sigmoid_expression.hpp"
#include "sum_expression.hpp"

namespace nnk
{

	template <class T>
	expression<T> softmax(const expression<T>& base)
	{
		expression<T> sig = sigmoid(base);
		expression<T> sum_sig = sum(sig);
		return sig / sum_sig;
	}

}
