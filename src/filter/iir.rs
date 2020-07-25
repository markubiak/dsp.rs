/// Basic implementations of common discrete filters
use arraydeque::{ArrayDeque, Wrapping};
use generic_array::typenum::U3;
use generic_array::{ArrayLength, GenericArray};
use itertools::izip;

use super::Filter;

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
pub type BiquadFilter = IIRFilter<U3>;

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
}

impl<N: ArrayLength<f32>> Filter for IIRFilter<N> {
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
}

/// ------------------------------------------------------------------------------------------------
/// Module unit tests
/// ------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::ForwardFFT;
    use assert_approx_eq::assert_approx_eq;
    use generic_array::typenum::U9;
    use std::f32::consts::PI;

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
        for (analog_samp, digital_samp) in izip!(analog_response, digital_response) {
            assert_approx_eq!(analog_samp, digital_samp, 0.05);
        }
    }

    #[test]
    fn test_iir_lowpass_real() {
        // Test our IIR filter implementation by filtering out from
        // a simulated signal.
        // An 8th order Butterworth recursive filter with fc = Pi/3
        // was generated using Scipy's signal.butter() funtion, and is
        // applied to a composit signal consisting of equal-amplitude
        // components with frequencies Pi/10 and Pi/2. The filtered
        // result should contain only the Pi/10 frequency component.
        let b = vec![
            7.092397e-04,
            5.673917e-03,
            1.985871e-02,
            3.971742e-02,
            4.964678e-02,
            3.971742e-02,
            1.985871e-02,
            5.673917e-03,
            7.092397e-04,
        ];
        let a = vec![
            1.000000e+00,
            -2.652714e+00,
            3.914857e+00,
            -3.608835e+00,
            2.259673e+00,
            -9.570251e-01,
            2.663485e-01,
            -4.404093e-02,
            3.300823e-03,
        ];
        let f1 = 1.0 / 10.0;
        let f2 = 1.0 / 2.0;

        // Generate our composite test input signal.
        let mut x = vec![0.0; 1024];
        for i in 0..x.len() {
            x[i] =
                f32::cos(2.0 * PI * f1 * (i as f32)) + 2.0 * f32::cos(2.0 * PI * f2 * (i as f32));
        }

        // Create and run our high-order IIR filter to remove f2.
        let mut lpf: IIRFilter<U9> = IIRFilter::new(&b, &a);
        let mut y = vec![0.0; 1024];
        lpf.process(&x, &mut y);

        // Check our result against a pure wave of frequency f1 by checking
        // that their frequency-domain representations have energy at
        // the same bin index.
        let mut fft = ForwardFFT::new(2048);
        let f1_bin = f32::round(f1 * 2048.0) as i32;
        y.extend(vec![0.0; 1024]); // Zero-pad our signal
        let fft_y = fft.process_real(&y);
        // Handwritten argmax as f32 does not implement Ord. Nice.
        let fft_argmax =
            fft_y.iter().enumerate().fold(
                0,
                |argmax, (idx, val)| {
                    if *val > fft_y[argmax] {
                        idx
                    } else {
                        argmax
                    }
                },
            ) as i32;

        // The filter should have removed f2, leaving the bin corresponding
        // to f1 with peak frequency-domain amplitude
        assert_eq!(f1_bin, fft_argmax);
    }
}
