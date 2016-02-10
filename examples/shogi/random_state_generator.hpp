#pragma once
#include <nanikanizer/nanikanizer.hpp>
#include "game.hpp"
#include "enumerate_action.hpp"
#include "apply_state.hpp"

namespace shogi
{

	class random_state_generator
	{
	public:

		explicit random_state_generator(double sample_rate = 0.001)
			: sample_rate_(sample_rate)
		{
		}

		void next()
		{
			action_buffer_.clear();
			enumerate_action(game_, std::back_inserter(action_buffer_));

			std::uniform_int<std::size_t> distribution(0, action_buffer_.size() - 1);
			std::size_t index = distribution(generator_);
			const action& act = action_buffer_[index];

			game_.apply(act);

			if (game_.state().is_finished())
				game_.reset();
		}

		void randomize()
		{
			while (true)
			{
				next();

				if (std::uniform_real<>(0.0, 1.0)(generator_) < sample_rate_)
					break;
			}
		}

		const game& get() const
		{
			return game_;
		}

		void generate(float* data)
		{
			randomize();
			apply_state(data, game_.state(), game_.turn());
		}

		void generate(float* data, std::size_t size)
		{
			for (std::size_t i = 0; i < size; ++i)
			{
				generate(data);
				data += input_size;
			}
		}

	private:

		double sample_rate_;

		std::mt19937 generator_;
		std::vector<action> action_buffer_;

		game game_;

	};

}
