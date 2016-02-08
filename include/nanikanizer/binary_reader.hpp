#pragma once

namespace nnk
{

	class binary_reader : boost::noncopyable
	{
	public:

		explicit binary_reader(std::istream& stream)
			: stream_(stream)
		{
		}

		void read(char& value)
		{
			std::int64_t temp;
			read(temp);
			value = boost::numeric_cast<char>(temp);
		}

		void read(std::int8_t& value)
		{
			std::int64_t temp;
			read(temp);
			value = boost::numeric_cast<std::int8_t>(temp);
		}

		void read(std::int16_t& value)
		{
			std::int64_t temp;
			read(temp);
			value = boost::numeric_cast<std::int16_t>(temp);
		}

		void read(std::int32_t& value)
		{
			std::int64_t temp;
			read(temp);
			value = boost::numeric_cast<std::int32_t>(temp);
		}

		void read(std::int64_t& value)
		{
			read_impl(value);
		}

		void read(std::uint8_t& value)
		{
			std::uint64_t temp;
			read(temp);
			value = boost::numeric_cast<std::uint8_t>(temp);
		}

		void read(std::uint16_t& value)
		{
			std::uint64_t temp;
			read(temp);
			value = boost::numeric_cast<std::uint16_t>(temp);
		}

		void read(std::uint32_t& value)
		{
			std::uint64_t temp;
			read(temp);
			value = boost::numeric_cast<std::uint32_t>(temp);
		}

		void read(std::uint64_t& value)
		{
			read_impl(value);
		}

		void read(float& value)
		{
			read_impl(value);
		}

		void read(double& value)
		{
			read_impl(value);
		}

		template <class T>
		void read(std::valarray<T>& value)
		{
			std::size_t size;
			read(size);

			value.resize(size);

			for (std::size_t i = 0; i < size; ++i)
				read(value[i]);
		}

	private:

		template <class T>
		void read_impl(T& value)
		{
			stream_.read(reinterpret_cast<char*>(&value), sizeof(value));
		}

		std::istream& stream_;

	};

}
