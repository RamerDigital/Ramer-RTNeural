#pragma once

namespace RTNEURAL_NAMESPACE
{

/**
 * For templated recurrent layers (e.g. LSTMLayerT, GRULayerT),
 * this class can be used as a template argument to allow the
 * recurrent layer to perform real-time sample-rate correction.
 *
 * For example, if you have a GRU network that was trained at 48 kHz
 * and want to process data at 96 kHz, you could enable sample-rate
 * correction for that layer, and prepare it to use a 2-sample delay,
 * instead of the standard 1-sample delay (since the target sample rate
 * is 2x the training sample rate). Note that sample-rate correction
 * does not support delay lengths less than 1-sample, so the target sample
 * rate must always be greater than or equal to the training sample rate.
 */
enum class SampleRateCorrectionMode
{
    None, // no sample rate correction
    NoInterp, // sample rate correction with no interpolation (only appropriate for integer delay lengths)
    LinInterp, // sample rate correction with linear interpolation (can be used with non-integer delay lengths)
};

/** Divides two numbers and rounds up if there is a remainder. */
template <typename T>
constexpr T ceil_div(T num, T den)
{
    return (num + den - 1) / den;
}

struct Empty
{
};
} // namespace RTNEURAL_NAMESPACE

#if RTNEURAL_USE_EIGEN
#include <Eigen/Dense>

namespace RTNEURAL_NAMESPACE
{
#if RTNEURAL_DEFAULT_ALIGNMENT == 32
constexpr auto RTNeuralEigenAlignment = Eigen::Aligned32;
#elif RTNEURAL_DEFAULT_ALIGNMENT == 16
constexpr auto RTNeuralEigenAlignment = Eigen::Aligned16;
#elif RTNEURAL_DEFAULT_ALIGNMENT == 8
constexpr auto RTNeuralEigenAlignment = Eigen::Aligned8;
#else
#error "Unsupported alignment"
#endif
} // namespace RTNEURAL_NAMESPACE

#elif RTNEURAL_USE_XSIMD
#include <xsimd/xsimd.hpp>
#include <array>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace xsimd
{
    template <class Arch = default_arch, class I1, class I2, class O1, class UF>
    void transform(I1 first, I2 last, O1 out_first, UF&& f) noexcept
    {
        using value_type = typename std::decay<decltype(*first)>::type;
        using batch_type = batch<value_type, Arch>;

        std::size_t size = static_cast<std::size_t>(std::distance(first, last));
        std::size_t simd_size = batch_type::size;

        const auto* ptr_begin = &(*first);
        auto* ptr_out = &(*out_first);

        std::size_t align_begin = xsimd::get_alignment_offset(ptr_begin, size, simd_size);
        std::size_t out_align = xsimd::get_alignment_offset(ptr_out, size, simd_size);
        std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        if (align_begin == out_align)
        {
            for (std::size_t i = 0; i < align_begin; ++i)
            {
                out_first[i] = f(first[i]);
            }

            for (std::size_t i = align_begin; i < align_end; i += simd_size)
            {
                batch_type batch = batch_type::load_aligned(&first[i]);
                xsimd::store_aligned(&out_first[i], f(batch));
            }

            for (std::size_t i = align_end; i < size; ++i)
            {
                out_first[i] = f(first[i]);
            }
        }
        else
        {
            for (std::size_t i = 0; i < align_begin; ++i)
            {
                out_first[i] = f(first[i]);
            }

            for (std::size_t i = align_begin; i < align_end; i += simd_size)
            {
                batch_type batch = batch_type::load_aligned(&first[i]);
                xsimd::store_unaligned(&out_first[i], f(batch));
            }

            for (std::size_t i = align_end; i < size; ++i)
            {
                out_first[i] = f(first[i]);
            }
        }
    }

    template <class Arch = default_arch, class I1, class I2, class I3, class O1, class UF>
    void transform(I1 first_1, I2 last_1, I3 first_2, O1 out_first, UF&& f) noexcept
    {
        using value_type = typename std::decay<decltype(*first_1)>::type;
        using batch_type = batch<value_type, Arch>;

        std::size_t size = static_cast<std::size_t>(std::distance(first_1, last_1));
        std::size_t simd_size = batch_type::size;

        const auto* ptr_begin_1 = &(*first_1);
        const auto* ptr_begin_2 = &(*first_2);
        auto* ptr_out = &(*out_first);

        std::size_t align_begin_1 = xsimd::get_alignment_offset(ptr_begin_1, size, simd_size);
        std::size_t align_begin_2 = xsimd::get_alignment_offset(ptr_begin_2, size, simd_size);
        std::size_t out_align = xsimd::get_alignment_offset(ptr_out, size, simd_size);
        std::size_t align_end = align_begin_1 + ((size - align_begin_1) & ~(simd_size - 1));

#define XSIMD_LOOP_MACRO(A1, A2, A3)                                   \
    for (std::size_t i = 0; i < align_begin_1; ++i)                    \
    {                                                                  \
        out_first[i] = f(first_1[i], first_2[i]);                      \
    }                                                                  \
                                                                       \
    batch_type batch_1, batch_2;                                       \
    for (std::size_t i = align_begin_1; i < align_end; i += simd_size) \
    {                                                                  \
        batch_1 = batch_type::A1(&first_1[i]);                         \
        batch_2 = batch_type::A2(&first_2[i]);                         \
        xsimd::A3(&out_first[i], f(batch_1, batch_2));                 \
    }                                                                  \
                                                                       \
    for (std::size_t i = align_end; i < size; ++i)                     \
    {                                                                  \
        out_first[i] = f(first_1[i], first_2[i]);                      \
    }

        if (align_begin_1 == out_align && align_begin_1 == align_begin_2)
        {
            XSIMD_LOOP_MACRO(load_aligned, load_aligned, store_aligned);
        }
        else if (align_begin_1 == out_align && align_begin_1 != align_begin_2)
        {
            XSIMD_LOOP_MACRO(load_aligned, load_unaligned, store_aligned);
        }
        else if (align_begin_1 != out_align && align_begin_1 == align_begin_2)
        {
            XSIMD_LOOP_MACRO(load_aligned, load_aligned, store_unaligned);
        }
        else if (align_begin_1 != out_align && align_begin_1 != align_begin_2)
        {
            XSIMD_LOOP_MACRO(load_aligned, load_unaligned, store_unaligned);
        }

#undef XSIMD_LOOP_MACRO
    }

    namespace detail
    {
        struct plus
        {
            template <class X, class Y>
            auto operator()(X&& x, Y&& y) noexcept -> decltype(x + y) { return x + y; }
        };
    }

    template <class Arch = default_arch, class Iterator1, class Iterator2, class Init, class BinaryFunction = detail::plus>
    Init reduce(Iterator1 first, Iterator2 last, Init init, BinaryFunction&& binfun = detail::plus {}) noexcept
    {
        using value_type = typename std::decay<decltype(*first)>::type;
        using batch_type = batch<value_type, Arch>;

        std::size_t size = static_cast<std::size_t>(std::distance(first, last));
        constexpr std::size_t simd_size = batch_type::size;

        if (size < simd_size)
        {
            while (first != last)
            {
                init = binfun(init, *first++);
            }
            return init;
        }

        const auto* const ptr_begin = &(*first);

        std::size_t align_begin = xsimd::get_alignment_offset(ptr_begin, size, simd_size);
        std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        for (std::size_t i = 0; i < align_begin; ++i)
        {
            init = binfun(init, first[i]);
        }

        auto ptr = ptr_begin + align_begin;
        batch_type batch_init = batch_type::load_aligned(ptr);
        ptr += simd_size;
        for (auto const end = ptr_begin + align_end; ptr < end; ptr += simd_size)
        {
            batch_type batch = batch_type::load_aligned(ptr);
            batch_init = binfun(batch_init, batch);
        }

        alignas(batch_type) std::array<value_type, simd_size> arr;
        xsimd::store_aligned(arr.data(), batch_init);
        for (auto x : arr)
            init = binfun(init, x);

        for (std::size_t i = align_end; i < size; ++i)
        {
            init = binfun(init, first[i]);
        }

        return init;
    }
}

namespace RTNEURAL_NAMESPACE
{

template <typename T>
static inline xsimd::simd_type<T> set_value(xsimd::simd_type<T> x, int idx, T value) noexcept
{
    union UnionType
    {
        xsimd::simd_type<T> v;
        T s[xsimd::simd_type<T>::size];
    };
    UnionType u { x };

    u.s[idx] = value;
    return u.v;
}

template <typename T>
static inline T get_value(xsimd::simd_type<T> x, int idx) noexcept
{
    union UnionType
    {
        xsimd::simd_type<T> v;
        T s[xsimd::simd_type<T>::size];
    };
    UnionType u { x };

    return u.s[idx];
}

template <typename T>
static inline T vMult(const T* arg1, const T* arg2, T* prod,
    int dim) noexcept
{
    xsimd::transform(arg1, &arg1[dim], arg2, prod,
        [](auto const& a, auto const& b)
        { return a * b; });

    return xsimd::reduce(prod, &prod[dim], (T)0);
}

template <typename T>
static inline void vAdd(const T* in1, const T* in2, T* out,
    int dim) noexcept
{
    xsimd::transform(in1, &in1[dim], in2, out,
        [](auto const& a, auto const& b)
        { return a + b; });
}

template <typename T>
static inline void vSub(const T* in1, const T* in2, T* out,
    int dim) noexcept
{
    xsimd::transform(in1, &in1[dim], in2, out,
        [](auto const& a, auto const& b)
        { return a - b; });
}

template <typename T>
static inline void vProd(const T* in1, const T* in2, T* out,
    int dim) noexcept
{
    xsimd::transform(in1, &in1[dim], in2, out,
        [](auto const& a, auto const& b)
        { return a * b; });
}

template <typename T>
static inline void vCopy(const T* in, T* out, int dim) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type vec = xsimd::load_aligned(&in[i]);
        xsimd::store_aligned(&out[i], vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
        out[i] = in[i];
}

template <typename T, typename MathsProvider>
static inline void sigmoid(const T* in, T* out, int dim) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&in[i]);
        b_type y_vec = MathsProvider::sigmoid(x_vec);
        xsimd::store_aligned(&out[i], y_vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
        out[i] = MathsProvider::sigmoid(in[i]);
}

template <typename T, typename MathsProvider>
static inline void softmax(const T* in, T* out, int dim) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    b_type exp_sum_vec {};

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&in[i]);
        b_type y_vec = MathsProvider::exp(x_vec);
        exp_sum_vec += y_vec;
        xsimd::store_aligned(&out[i], y_vec);
    }

    T exp_sum = xsimd::reduce_add(exp_sum_vec);

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
    {
        out[i] = MathsProvider::exp(in[i]);
        exp_sum += out[i];
    }

    const auto exp_sum_recip = (T)1 / exp_sum;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&out[i]);
        b_type y_vec = x_vec * exp_sum_recip;
        xsimd::store_aligned(&out[i], y_vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
    {
        out[i] *= exp_sum_recip;
    }
}

template <typename T, typename MathsProvider>
static inline void tanh(const T* in, T* out, int dim) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&in[i]);
        b_type y_vec = MathsProvider::tanh(x_vec);
        xsimd::store_aligned(&out[i], y_vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
        out[i] = MathsProvider::tanh(in[i]);
}

template <typename T, typename MathsProvider>
static inline void elu(const T* in, T* out, int dim, T alpha) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&in[i]);
        b_type y_vec = xsimd::select(x_vec > (T)0, x_vec, alpha * (MathsProvider::exp(x_vec) - (T)1));
        xsimd::store_aligned(&out[i], y_vec);
    }

    // Remaining part that cannot be vectorized
    for(auto i = vec_size; i < dim; ++i)
        out[i] = in[i] > (T)0 ? in[i] : (alpha * (MathsProvider::exp(in[i]) - (T)1));
}
} // namespace RTNEURAL_NAMESPACE

#else // STL backend
#include <algorithm>
#include <cmath>
#include <numeric>

namespace RTNEURAL_NAMESPACE
{
template <typename T>
static inline T vMult(const T* arg1, const T* arg2, int dim) noexcept
{
    return std::inner_product(arg1, arg1 + dim, arg2, (T)0);
}
} // namespace RTNEURAL_NAMESPACE

#endif
