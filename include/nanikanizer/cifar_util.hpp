#pragma once

namespace nnk
{
	namespace cifar_10
	{

		typedef std::pair<int, std::valarray<float>> tagged_image;

		static const std::size_t channel_size = 32 * 32;
		static const std::size_t whole_size = channel_size * 3;

		template <class Iterator>
		void load_images(const std::string& file, bool normalize, Iterator it)
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

				if (normalize)
				{
					values -= values.sum() / values.size();
					values /= (values * values).sum() / values.size();
				}

				*it++ = tagged_image(type, values);
			}
		}

	}
}
