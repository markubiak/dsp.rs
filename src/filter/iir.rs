/// Basic implementations of common discrete filters
use arraydeque::{ArrayDeque, Wrapping};
use generic_array::typenum::U3;
use generic_array::{ArrayLength, GenericArray};
use itertools::izip;
use crate::RealBuffer;


/// A generic IIR filter with matching a/b tap lengths
#[derive(Clone, Debug)]
pub struct IIRFilter<N: ArrayLength<f32>> {
    x: ArrayDeque<GenericArray<f32, N>, Wrapping>,
    y: ArrayDeque<GenericArray<f32, N>, Wrapping>,
    b: GenericArray<f32, N>,
    a: GenericArray<f32, N>,
}

/// A biquad IIR filter common for second-order section
/// implementations
#[allow(dead_code)]
type BiquadFilter = IIRFilter<U3>;


impl<N: ArrayLength<f32>> IIRFilter<N> {
    /// Returns a new biquad IIR filter. Failure if a/b not correct lengths
    pub fn new(b: &[f32], a: &[f32]) -> IIRFilter<N> {
        // Sanity check
        assert_ne!(a[0], 0.0); // a0 of 0 results in divide by 0

        // Initialize sample histories
        let mut x: ArrayDeque<GenericArray<f32, N>, Wrapping> = ArrayDeque::new();
        assert_eq!(b.len(), x.capacity());
        for _ in 0..x.capacity() {
            x.push_front(0.0);
        }

        let mut y: ArrayDeque<GenericArray<f32, N>, Wrapping> = ArrayDeque::new();
        assert_eq!(a.len(), y.capacity());
        for _ in 0..y.capacity() {
            y.push_front(0.0);
        }

        // Clone the b coefficients from passed in slice
        let b_arr = GenericArray::clone_from_slice(b);

        // Clone the a coefficients, inverting a[1..] by
        // the definition of an IIR filter
        let mut a_arr = GenericArray::clone_from_slice(a);
        for i in 1..a_arr.len() {
            a_arr[i] = -a_arr[i];
        }

        // New filter with x/y initalized to same length as a/b
        IIRFilter {
            x,
            y,
            b: b_arr,
            a: a_arr,
        }
    }

    /// Process one sample of the input signal and returns one sample of the
    /// output signal.
    fn process_one(&mut self, in_samp: f32) -> f32 {
        // Shift in old values
        self.x.pop_back();
        self.x.push_front(in_samp.clone());
        self.y.pop_back();

        // Compute the filter result
        let mut sum = 0.0;
        for (xi, bi) in izip!(self.x.iter(), self.b.iter()) {
            sum += *xi * *bi;
        }
        for (yi, ai) in izip!(self.y.iter(), self.a[1..].iter()) {
            sum += *yi * *ai;
        }
        sum /= self.a[0];

        // Update y and return the result
        self.y.push_front(sum);
        sum
    }

    /// Processes in_slice as a slice of samples as inputs to the filter,
    /// writing results to out_slice.
    pub fn process(&mut self, input: &RealBuffer, output: &mut RealBuffer) {
        let size = std::cmp::min(input.len(), output.len());
        (0..size).for_each(|i| output[i] = self.process_one(input[i]));
    }
}

/// ------------------------------------------------------------------------------------------------
/// Module unit tests
/// ------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_biquad_bilinear_rc() {
        // Test our biquad by using the bilinear transform to create
        // a digital filter with similar response to a basic analog RC
        // filter with constants T=0.1s RC=1.
        // Generate and compare their step responses over 5 seconds.
        let rc = 1.0;
        let t_samp = 0.1;

        // Analog step response evaluated every T seconds.
        let mut analog_response = Vec::new();
        for i in 0..50 {
            let t = (i as f32) * t_samp;
            analog_response.push(1.0 - f32::exp(-t / rc));
        }

        // Compute and run the equivalent digital filter.
        // Bilinear transform of RC filter:
        // https://en.wikipedia.org/wiki/Bilinear_transform#Example
        let b = [1.0, 1.0, 0.0];
        let a = [1.0 + (2.0 * rc / t_samp), 1.0 - (2.0 * rc / t_samp), 0.0];
        let mut biquad_rc: BiquadFilter = IIRFilter::new(&b, &a);
        let dig_unit_step = vec![1.0; 50];
        let mut digital_response = vec![0.0; 50];
        biquad_rc.process(&dig_unit_step, &mut digital_response);

        // Compare the filter unit step responses. Since there
        // is some difference between the initial state of the
        // filters, use a less aggressive 5% threshold
        for i in 0..50 {
            println!("{}", i);
            assert_approx_eq!(analog_response[i], digital_response[i], 0.05);
        }
    }
}
