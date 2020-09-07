//! Implementations of common discrete-time filtering operations
use itertools::izip;

use crate::RealBuffer;

pub mod fir;
pub mod iir;

/// Defines shared behavior for all filter implementations.
pub trait Filter {
    /// Process a single real sample.
    fn process_one(&mut self, in_samp: f32) -> f32;

    /// Processes a slice of samples.
    /// The default implementation simply calls process_one for each input sample,
    /// which is sufficient for most implementations.
    fn process(&mut self, input: &RealBuffer, output: &mut RealBuffer) {
        assert_eq!(input.len(), output.len());
        for (in_samp, out_samp) in izip!(input.iter(), output.iter_mut()) {
            *out_samp = self.process_one(*in_samp);
        }
    }
}
