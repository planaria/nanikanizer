#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <random>
#include <nanikanizer/nanikanizer.hpp>

typedef std::pair<int, std::valarray<float>> image_type;
static const std::size_t channel_size = 32 * 32;
static const std::size_t whole_size = channel_size * 3;

template <class Iterator>
void load_images(const std::string& file, Iterator it)
{
	std::ifstream is(file, std::ios_base::binary);

	std::vector<char> buffer(whole_size);
	std::valarray<float> values(whole_size);

	while (true)
	{
		int type = is.get();
		if (is.eof())
			break;

		is.read(buffer.data(), whole_size);

		for (std::size_t i = 0; i < channel_size; ++i)
		{
			values[i * 3] = static_cast<float>(static_cast<unsigned char>(buffer[i]));
			values[i * 3 + 1] = static_cast<float>(static_cast<unsigned char>(buffer[i + channel_size]));
			values[i * 3 + 2] = static_cast<float>(static_cast<unsigned char>(buffer[i + channel_size * 2]));
		}

		values -= values.sum() / values.size();
		values /= (values * values).sum() / values.size();

		*it++ = image_type(type, values);
	}
}

int main(int /*argc*/, char* /*argv*/[])
{
	try
	{
		std::vector<std::string> data_files =
		{
			"data_batch_1.bin",
			"data_batch_2.bin",
			"data_batch_3.bin",
			"data_batch_4.bin",
			"data_batch_5.bin",
		};

		std::string test_file = "test_batch.bin";

		std::vector<image_type> data_images;
		std::vector<image_type> test_images;

		for (const std::string& file : data_files)
			load_images(file, std::back_inserter(data_images));

		load_images(test_file, std::back_inserter(test_images));

		std::size_t id_size = 10;

		std::vector<std::valarray<float>> ids(id_size);

		for (std::size_t i = 0; i < id_size; ++i)
		{
			std::valarray<float> id(id_size);
			id[i] = 1.0f;
			ids[i] = id;
		}

		typedef std::pair<std::string, std::shared_ptr<nnk::optimizer_base>> named_optimizer_type;

		std::vector<named_optimizer_type> optimizers =
		{
			//std::make_pair("SGD", std::make_shared<nnk::sgd_optimizer>()),
			//std::make_pair("AdaGrad", std::make_shared<nnk::adagrad_optimizer>()),
			//std::make_pair("RMSProp", std::make_shared<nnk::rmsprop_optimizer>()),
			std::make_pair("AdaDelta", std::make_shared<nnk::adadelta_optimizer>()),
			//std::make_pair("Adam", std::make_shared<nnk::adam_optimizer>()),
		};

		for (auto& key_value : optimizers)
		{
			std::ofstream os(key_value.first + ".log");

			std::cout << std::fixed << std::setprecision(5);
			os << std::fixed << std::setprecision(5);

			std::cout << key_value.first << std::endl;
			os << key_value.first << std::endl;

			nnk::linear_layer<float> l1(75, 6);
			nnk::linear_layer<float> l2(150, 16);
			nnk::linear_layer<float> l3(400, 120);
			nnk::linear_layer<float> l4(120, 84);
			nnk::linear_layer<float> l5(84, id_size);

			std::size_t batch_size = 900;

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
			auto x10 = nnk::softmax(l5(x9), id_size);
			auto loss = nnk::cross_entropy(x10 - y.expr());

			auto get_answer = [&](std::size_t index)
			{
				const auto& r = x10.root()->output();
				auto begin = &r[index * id_size];
				auto end = begin + id_size;
				auto it = std::max_element(begin, end);
				return it - begin;
			};

			nnk::evaluator<float> ev(loss);

			auto& optimizer = *key_value.second;

			optimizer.add_parameter(l1);
			optimizer.add_parameter(l2);
			optimizer.add_parameter(l3);
			optimizer.add_parameter(l4);
			optimizer.add_parameter(l5);

			std::mt19937 generator;
			std::uniform_int<std::size_t> index_generator(0, data_images.size() - 1);

			std::vector<int> answers(batch_size);

			for (std::size_t i = 0; i < 1000; ++i)
			{
				optimizer.zero_grads();

				for (std::size_t i = 0; i < batch_size; ++i)
				{
					std::size_t index = index_generator(generator);
					const image_type& image = data_images[index];

					answers[i] = image.first;

					for (std::size_t j = 0; j < whole_size; ++j)
						x1.value()[i * whole_size + j] = image.second[j];

					for (std::size_t j = 0; j < id_size; ++j)
						y.value()[i * id_size + j] = ids[image.first][j];
				}

				double loss_value = ev.forward()[0];
				ev.backward();

				std::size_t count_ok = 0;

				for (std::size_t i = 0; i < batch_size; ++i)
					if (get_answer(i) == answers[i])
						++count_ok;

				std::cout << loss_value << std::endl;
				os << loss_value << std::endl;

				optimizer.update();
			}

			std::size_t count_ok = 0;

			for (const auto& image : test_images)
			{
				x1.value() = image.second;

				ev.forward();

				if (get_answer(0) == image.first)
					++count_ok;
			}

			double rate = static_cast<double>(count_ok) / static_cast<double>(test_images.size());
			std::cout << "rate: " << rate << std::endl;
			os << "rate: " << rate << std::endl;
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
