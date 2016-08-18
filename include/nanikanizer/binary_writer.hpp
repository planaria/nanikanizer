#pragma once

namespace nnk
{

	class binary_writer : boost::noncopyable
	{
	public:

		explicit binary_writer(std::ostream& stream)
			: stream_(stream)
		{
		}

		void write(bool value)
		{
			write(value ? 1 : 0);
		}

		void write(char value)
		{
			write(static_cast<std::int64_t>(value));
		}

		void write(std::int8_t value)
		{
			write(static_cast<std::int64_t>(value));
		}

		void write(std::int16_t value)
		{
			write(static_cast<std::int64_t>(value));
		}

		void write(std::int32_t value)
		{
			write(static_cast<std::int64_t>(value));
		}

		void write(std::int64_t value)
		{
			write_impl(value);
		}

		void write(std::uint8_t value)
		{
			write(static_cast<std::uint64_t>(value));
		}

		void write(std::uint16_t value)
		{
			write(static_cast<std::uint64_t>(value));
		}

		void write(std::uint32_t value)
		{
			write(static_cast<std::uint64_t>(value));
		}

		void write(std::uint64_t value)
		{
			write_impl(value);
		}

		void write(float value)
		{
			write_impl(value);
		}

		void write(double value)
		{
			write_impl(value);
		}

		template <class T>
		void write(const std::valarray<T>& value)
		{
			write(value.size());
			for (std::size_t i = 0; i < value.size(); ++i)
				write(value[i]);
		}

	private:

		template <class T>
		void write_impl(T value)
		{
			stream_.write(reinterpret_cast<const char*>(&value), sizeof(value));
		}

		std::ostream& stream_;

	};

}
