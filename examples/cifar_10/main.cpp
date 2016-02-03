#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <nanikanizer/nanikanizer.hpp>

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

		std::vector<nnk::cifar_10::tagged_image> data_images;
		std::vector<nnk::cifar_10::tagged_image> test_images;

		for (const std::string& file : data_files)
			nnk::cifar_10::load_images(file, true, std::back_inserter(data_images));

		nnk::cifar_10::load_images(test_file, true, std::back_inserter(test_images));

		std::size_t id_size = 10;

		auto ids = nnk::make_ids(id_size, 0.0f, 1.0f);

		typedef std::pair<std::string, std::shared_ptr<nnk::optimizer_base>> named_optimizer_type;

		nnk::linear_layer<float> l1(27, 6);
		nnk::linear_layer<float> l2(54, 12);
		nnk::linear_layer<float> l3(108, 64);
		nnk::linear_layer<float> l4(576, 64);
		nnk::linear_layer<float> l5(576, 32);
		nnk::linear_layer<float> l6(288, 32);
		nnk::linear_layer<float> l7(32, 10);

		std::size_t batch_size = 1024;

		nnk::variable<float> x0;
		nnk::variable<float> y;

		nnk::expression<float> x = x0.expr();
		// 32 * 32 * 3
		x = nnk::convolution_2d(x, 32, 32, 3, 3, 3);
		// 30 * 30 * 27
		x = nnk::relu(l1.forward(x));
		// 30 * 30 * 6
		x = nnk::convolution_2d(x, 30, 30, 6, 3, 3);
		// 28 * 28 * 54
		x = nnk::relu(l2.forward(x));
		// 28 * 28 * 12
		x = nnk::max_pooling_2d(x, 28, 28, 12, 2, 2);
		// 14 * 14 * 12
		x = nnk::convolution_2d(x, 14, 14, 12, 3, 3);
		// 12 * 12 * 108
		x = nnk::relu(l3.forward(x));
		// 12 * 12 * 64
		x = nnk::convolution_2d(x, 12, 12, 64, 3, 3);
		// 10 * 10 * 576
		x = nnk::relu(l4.forward(x));
		// 10 * 10 * 64
		x = nnk::max_pooling_2d(x, 10, 10, 64, 2, 2);
		// 5 * 5 * 64
		x = nnk::convolution_2d(x, 5, 5, 64, 3, 3);
		// 3 * 3 * 576
		x = nnk::relu(l5.forward(x));
		// 3 * 3 * 32 = 288
		x = nnk::relu(l6.forward(x));
		// 32
		x = nnk::softmax(l7.forward(x), id_size);
		// 10

		auto reg =
			nnk::sum(nnk::abs(l1.weight().expr())) +
			nnk::sum(nnk::abs(l2.weight().expr())) +
			nnk::sum(nnk::abs(l3.weight().expr())) +
			nnk::sum(nnk::abs(l4.weight().expr())) +
			nnk::sum(nnk::abs(l5.weight().expr())) +
			nnk::sum(nnk::abs(l6.weight().expr())) +
			nnk::sum(nnk::abs(l7.weight().expr()));

		auto loss = nnk::cross_entropy(x - y.expr()) + reg * nnk::expression<float>({ 0.1f });

		auto get_answer = [&](std::size_t index)
		{
			const auto& r = x.root()->output();
			auto begin = &r[index * id_size];
			auto end = begin + id_size;
			auto it = std::max_element(begin, end);
			return it - begin;
		};

		nnk::evaluator<float> ev(loss);

		nnk::adam_optimizer optimizer;

		optimizer.add_parameter(l1);
		optimizer.add_parameter(l2);
		optimizer.add_parameter(l3);
		optimizer.add_parameter(l4);
		optimizer.add_parameter(l5);
		optimizer.add_parameter(l6);
		optimizer.add_parameter(l7);

		std::mt19937 generator;
		std::uniform_int<std::size_t> index_generator(0, data_images.size() - 1);

		std::cout << "Loss,Rate" << std::endl;
		std::cout << std::fixed << std::setprecision(5);

		for (std::size_t i = 0; i < 100; ++i)
		{
			x0.value().resize(batch_size * nnk::cifar_10::whole_size);
			y.value().resize(batch_size * id_size);

			float last_loss = 0.0f;

			for (std::size_t j = 0; j < 100; ++j)
			{
				for (std::size_t k = 0; k < batch_size; ++k)
				{
					std::size_t index = index_generator(generator);
					const auto& image = data_images[index];

					for (std::size_t l = 0; l < nnk::cifar_10::whole_size; ++l)
						x0.value()[k * nnk::cifar_10::whole_size + l] = image.second[l];

					for (std::size_t l = 0; l < id_size; ++l)
						y.value()[k * id_size + l] = ids[image.first][l];
				}

				optimizer.zero_grads();

				last_loss = ev.forward()[0];
				ev.backward();

				optimizer.update();
			}

			std::size_t count_ok = 0;

			for (const auto& image : test_images)
			{
				x0.value() = image.second;

				ev.forward();

				if (get_answer(0) == image.first)
					++count_ok;
			}

			double rate = static_cast<double>(count_ok) / static_cast<double>(test_images.size());
			std::cout << last_loss << "," << rate << std::endl;
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
