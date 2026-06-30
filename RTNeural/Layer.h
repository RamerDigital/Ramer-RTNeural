#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include <cstddef>
#include <string>
#include <concepts>
#include <span>

namespace RTNEURAL_NAMESPACE
{

/** Virtual base class for a generic neural network layer. */
template <std::floating_point T>
class Layer
{
public:
    /** Constructs a layer with given input and output size. */
    Layer(int in_size, int out_size)
        : in_size(in_size)
        , out_size(out_size)
    {
    }

    virtual ~Layer() = default;

    /** Returns the name of this layer. */
    virtual std::string getName() const noexcept { return ""; }

    /** Resets the state of this layer. */
    virtual void reset() { }

    /** Implements the forward propagation step for this layer. */
    virtual void forward(const T* input, T* out) noexcept = 0;

    /** Implements the forward propagation step using std::span for boundary safety. */
    virtual void forward(std::span<const T> input, std::span<T> out) noexcept
    {
        forward(input.data(), out.data());
    }

    const int in_size;
    const int out_size;
};

} // namespace RTNEURAL_NAMESPACE

#endif // LAYER_H_INCLUDED
