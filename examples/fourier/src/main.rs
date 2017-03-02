extern crate gnuplot;
extern crate num_complex;
extern crate dsp;

use gnuplot::{Figure, Color};
use dsp::signal::*;


fn main() {
    let xs = sample(&step, 0.0, 10.0, 1.);

    let idx: Vec<usize> = (0..xs.len()).collect();
    let ys: Vec<f32> = xs.iter().map(|x| x.re).collect();
    let mut fg = Figure::new();
    fg.axes2d().lines(&idx, &ys, &[Color("red")]);
    fg.show();
}