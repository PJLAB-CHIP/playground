#pragma once

#include <cassert>
#include <tuple>

#if __cplusplus < 202002L
    #include <array>
    #include <cstdint>
#endif

namespace playground
{
/**
 * @brief Compute the offset of a multi-dimensional array.
 *
 * @param args First half is the indexes, second half is the size of each
 *             dimension.
 * @return std::uint32_t The offset of the multi-dimensional array.
 *
 * @example computeOffset(1, 2, 3, 4, 5, 6) -> 3*1 + 2*6 + 1*6*5 = 45
 */
template <typename OffsetT, typename... ArgsT>
constexpr auto computeOffset(ArgsT... args) -> OffsetT
{
    constexpr std::size_t nArgs = sizeof...(ArgsT);
    constexpr std::size_t nDims = nArgs / 2;

    OffsetT offset = 0, stride = 1;

#if __cplusplus >= 202002L
    auto params = std::make_tuple(static_cast<OffsetT>(args)...);
    [&]<std::size_t... I>(std::index_sequence<I...>)
    {
        ((I < nDims ? (offset += std::get<nDims - 1 - I>(params) * stride,
                       stride *= std::get<nArgs - 1 - I>(params))
                    : 0),
         ...);
    }
    (std::make_index_sequence<nDims>{});
#else
    auto params = std::array{static_cast<OffsetT>(args)...};
    for (std::size_t i = 0; i < nDims; ++i) {
        offset += params[nDims - 1 - i] * stride;
        stride *= params[nArgs - 1 - i];
    }
#endif

    return offset;
}

}  // namespace playground
