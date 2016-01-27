#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <random>
#include <nanikanizer/nanikanizer.hpp>

int main(int /*argc*/, char* /*argv*/[])
{
	try
	{
		std::vector<std::string> input_files =
		{
			"data_batch_1.bin",
			"data_batch_2.bin",
			"data_batch_3.bin",
			"data_batch_4.bin",
			"data_batch_5.bin",
		};

		typedef std::pair<int, std::valarray<float>> image_type;

		std::vector<image_type> images;

		static const std::size_t channel_size = 32 * 32;
		static const std::size_t whole_size = channel_size * 3;

		std::vector<char> data(whole_size);
		std::valarray<float> values(whole_size);

		std::vector<char> buffer(8 * 1024 * 1024);

		for (const std::string& file : input_files)
		{
			std::ifstream is(file.c_str(), std::ios_base::binary);
			is.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

			while (true)
			{
				int type = is.get();
				if (is.eof())
					break;

				is.read(data.data(), whole_size);

				for (std::size_t i = 0; i < channel_size; ++i)
				{
					values[i * 3] = static_cast<float>(static_cast<unsigned char>(data[i]));
					values[i * 3 + 1] = static_cast<float>(static_cast<unsigned char>(data[i + channel_size]));
					values[i * 3 + 2] = static_cast<float>(static_cast<unsigned char>(data[i + channel_size * 2]));
				}

				values -= values.sum() / values.size();
				values /= (values * values).sum() / values.size();

				images.push_back(image_type(type, values));
			}
		}

		std::size_t id_size = 10;

		std::vector<std::valarray<float>> ids(id_size);

		for (std::size_t i = 0; i < id_size; ++i)
		{
			std::valarray<float> id(id_size);
			id[i] = 1.0f;
			ids[i] = id;
		}

		nnk::linear_layer<float> l1(75, 6);
		nnk::linear_layer<float> l2(150, 16);
		nnk::linear_layer<float> l3(400, 120);
		nnk::linear_layer<float> l4(120, 84);
		nnk::linear_layer<float> l5(84, id_size);

		std::size_t batch_size = 100;

		nnk::variable<float> x1(batch_size * whole_size);
		nnk::variable<float> y(batch_size * id_size);

		auto x2 = nnk::convolution_2d(x1.expr(), 32, 32, 3, 5, 5);
		auto x3 = nnk::relu(l1(x2));
		auto x4 = nnk::max_pooling_2d(x3, 28, 28, 6, 2, 2);
		auto x5 = nnk::convolution_2d(x4, 14, 14, 6, 5, 5);
		auto x6 = nnk::relu(l2(x5));
		auto x7 = nnk::max_pooling_2d(x6, 10, 10, 16, 2, 2);
		auto x8 = nnk::relu(l3(x7));
		auto x9 = nnk::relu(l4(x8));
		auto x10 = nnk::softmax(sigmoid(l5(x9)), id_size);
		auto loss = nnk::cross_entropy(x10 - y.expr());

		nnk::evaluator<float> ev(loss);

		nnk::adam_optimizer optimizer;
		optimizer.add_parameter(l1);
		optimizer.add_parameter(l2);
		optimizer.add_parameter(l3);
		optimizer.add_parameter(l4);
		optimizer.add_parameter(l5);

		std::mt19937 generator;
		std::uniform_int<std::size_t> index_generator(0, images.size() - 1);

		std::vector<int> answers(batch_size);

		while (true)
		{
			optimizer.zero_grads();

			for (std::size_t i = 0; i < batch_size; ++i)
			{
				std::size_t index = index_generator(generator);
				const image_type& image = images[index];

				answers[i] = image.first;

				for (std::size_t j = 0; j < whole_size; ++j)
					x1.value()[i * whole_size + j] = image.second[j];

				for (std::size_t j = 0; j < id_size; ++j)
					y.value()[i * id_size + j] = ids[image.first][j];
			}

			double loss_value = ev.forward()[0];
			ev.backward();

			const auto& r = x10.root()->output();

			std::size_t count_ok = 0;
			std::size_t count_ng = 0;

			for (std::size_t i = 0; i < batch_size; ++i)
			{
				auto begin = &r[i * id_size];
				auto end = begin + id_size;
				auto it = std::max_element(begin, end);
				std::size_t ans = it - begin;

				if (ans == answers[i])
					++count_ok;
				else
					++count_ng;
			}

			double rate = static_cast<double>(count_ok) / static_cast<double>(count_ok + count_ng);
			std::cout << std::fixed << std::setprecision(5) << loss_value << " " << rate << std::endl;

			optimizer.update();
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
