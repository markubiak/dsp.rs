/// Basic implementation of convolution operation via FIR filter
use arraydeque::{ArrayDeque, Wrapping};
use generic_array::{ArrayLength, GenericArray};
use itertools::izip;
use crate::RealBuffer;


/// A Finite Impulse Response (FIR) filter
#[derive(Clone,Debug)]
pub struct FIRFilter<N: ArrayLength<f32>> {
    x: ArrayDeque<GenericArray<f32, N>, Wrapping>,
    b: GenericArray<f32, N>
}


impl<N: ArrayLength<f32>> FIRFilter<N> {

    /// Returns a new FIRFilter from coefficients b
    pub fn new(b: &[f32]) -> FIRFilter<N> {

        // Initialize sample history
        let mut x: ArrayDeque<GenericArray<f32, N>, Wrapping> = ArrayDeque::new();
        assert_eq!(b.len(), x.capacity());
        for _ in 0..x.capacity() {
            x.push_front(0.0);
        }

        // Clone the b coefficients from passed in slice
        let b = GenericArray::clone_from_slice(b);

        // Return our new FIR filter
        FIRFilter { x, b }
    }

    /// Process one sample of the input signal and returns one sample of the
    /// output signal.
    fn process_one(&mut self, in_samp: f32) -> f32 {

        // Shift in old values
        self.x.pop_back();
        self.x.push_front(in_samp.clone());

        // Compute the filter result
        let mut sum = 0.0;
        for (xi, bi) in izip!(self.x.iter(), self.b.iter()) {
            sum += *xi * *bi;
        }

        // Return our calculated result
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
    use crate::window;
    use generic_array::typenum::{U5};

    #[test]
    fn test_fir_convolution() {
        // Test our FIR filter, which performs a discrete convolution
        // operation, by convolving a triangular signal with multiple
        // Kronecker delta functions. Check that each delta corresponds
        // to a shifted copy of the original triangle wave.
        let tri = window::triangular(5, 1, 5);
        let mut deltas = vec![0.0; 32];
        deltas[0] = 1.0;
        deltas[10] = 1.0;
        deltas[20] = 2.0; // double the amplitude on the last one

        // Compute and run the convolution operation via FIR filter.
        // Use the triangular sequence as the filter taps and
        // the pulse train (deltas) as the input sequence.
        let mut fir: FIRFilter<U5> = FIRFilter::new(&tri.samples);
        let mut conv_output = vec![0.0; 32];
        fir.process(&deltas, &mut conv_output);

        // Check each slice against the input window
        let zeros = [0.0; 5];
        assert_eq!(tri.samples[..], conv_output[0..5]);
        assert_eq!(zeros, conv_output[5..10]);
        assert_eq!(tri.samples[..], conv_output[10..15]);
        assert_eq!(zeros, conv_output[15..20]);
        let mut tri_doubled = [0.0; 5];
        for i in 0..5 {
            tri_doubled[i] = tri.samples[i] * 2.0;
        }
        assert_eq!(tri_doubled, conv_output[20..25]);
    }
}
