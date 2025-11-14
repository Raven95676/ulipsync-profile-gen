use rustfft::{num_complex::Complex32, FftPlanner};
use std::cell::RefCell;
use std::f32::consts::PI;

#[inline]
fn get_max_value(slice: &[f32]) -> f32 {
  slice.iter().map(|&sample| sample.abs()).fold(0.0, f32::max)
}

pub fn normalize(data: &mut [f32], peak: f32) {
  let m = get_max_value(data);
  if m > f32::EPSILON {
    let r = peak / m;
    for x in data.iter_mut() {
      *x *= r;
    }
  }
}

fn low_pass_filter_kernel(data: &mut [f32], cutoff: f32, tmp: &[f32], b: &mut [f32]) {
  let blen = b.len();

  for (i, b_val) in b.iter_mut().enumerate() {
    let x = i as f32 - (blen as f32 - 1.0) * 0.5;
    let ang = 2.0 * PI * cutoff * x;
    *b_val = 2.0 * cutoff * ang.sin() / ang;
  }

  let len = data.len();
  for i in 0..len {
    for j in 0..blen {
      if i >= j {
        data[i] += b[j] * tmp[i - j];
      }
    }
  }
}

pub fn low_pass_filter(data: &mut [f32], sample_rate: f32, cutoff: f32, range: f32) {
  let cutoff_n = (cutoff - range) / sample_rate;
  let range_n = range / sample_rate;

  let tmp = data.to_vec();

  let mut n = (3.1 / range_n).round_ties_even() as i32;

  if ((n + 1) % 2) == 0 {
    n += 1;
  }

  let blen = if n > 0 { n as usize } else { 0 };
  let mut b = vec![0.0; blen];

  low_pass_filter_kernel(data, cutoff_n, &tmp, &mut b);
}

pub fn downsample(input: &[f32], sample_rate: u32, target_sample_rate: u32, out: &mut Vec<f32>) {
  out.clear();
  if sample_rate <= target_sample_rate {
    out.extend_from_slice(input);
    return;
  }

  if sample_rate.is_multiple_of(target_sample_rate) {
    let skip = (sample_rate / target_sample_rate) as usize;
    let out_len = input.len() / skip;
    out.reserve(out_len.saturating_sub(out.capacity()));
    for i in 0..out_len {
      out.push(input[i * skip]);
    }
    return;
  }

  let df = (sample_rate as f32) / (target_sample_rate as f32);
  let out_len = (input.len() as f32 / df).round_ties_even() as usize;
  out.reserve(out_len.saturating_sub(out.capacity()));
  for j in 0..out_len {
    let f_index = df * (j as f32);
    let i0 = f_index.floor() as usize;
    // 感觉这里应该是i0+1，但是原始代码是这么写的
    let i1 = i0.min(input.len().saturating_sub(1));
    let t = f_index - (i0 as f32);
    let y = input[i0] * (1.0 - t) + input[i1] * t;
    out.push(y);
  }
}

pub fn pre_emphasis(data: &mut [f32], coeff: f32) {
  for i in (1..data.len()).rev() {
    data[i] -= coeff * data[i - 1]
  }
}

pub fn hamming(data: &mut [f32]) {
  let n = data.len() as f32;
  for (i, x) in data.iter_mut().enumerate() {
    let i = i as f32 / (n - 1.0);
    let w = 0.54 - 0.46 * (2.0 * PI * i).cos();
    *x *= w;
  }
}

// 理论上没问题，偷个懒（
pub fn fft(data: &[f32], complex: &mut Vec<Complex32>, out: &mut Vec<f32>) {
  let n = data.len();
  thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
  }
  complex.clear();
  complex.reserve(n.saturating_sub(complex.capacity()));
  for &x in data {
    complex.push(Complex32::new(x, 0.0));
  }
  if n > 0 {
    FFT_PLANNER.with(|planner_ref| {
      let mut planner = planner_ref.borrow_mut();
      let fft = planner.plan_fft_forward(n);
      fft.process(complex);
    });
  }
  out.clear();
  out.reserve(n.saturating_sub(out.capacity()));
  out.extend(complex.iter().map(|c| c.norm()));
}

#[inline]
pub fn power_to_db(array: &mut [f32]) {
  for value in array.iter_mut() {
    *value = 10.0 * value.log10();
  }
}

#[inline]
pub fn to_mel(hz: f32, slaney: bool) -> f32 {
  let a = if slaney { 2595.0 } else { 1127.0 };
  a * (hz / 700.0 + 1.0).ln()
}

#[inline]
pub fn to_hz(mel: f32, slaney: bool) -> f32 {
  let a = if slaney { 2595.0 } else { 1127.0 };
  700.0 * ((mel / a).exp() - 1.0)
}

pub fn dct(spectrum: &[f32], out: &mut [f32]) {
  let len = spectrum.len();
  let a = PI / len as f32;

  for (i, cep_val) in out.iter_mut().enumerate().take(len) {
    let mut sum = 0.0;
    for (j, spec_val) in spectrum.iter().enumerate() {
      let ang = (j as f32 + 0.5) * i as f32 * a;
      sum += *spec_val * ang.cos();
    }
    *cep_val = sum;
  }
}

pub fn mel_filter_bank(spectrum: &[f32], sample_rate: f32, mel_div: usize, out: &mut [f32]) {
  let len = spectrum.len();

  let f_max = sample_rate / 2.0;
  let mel_max = to_mel(f_max, false);
  let n_max = len / 2;
  let df = f_max / n_max as f32;
  let d_mel = mel_max / (mel_div + 1) as f32;

  for (n, out_val) in out.iter_mut().enumerate().take(mel_div) {
    let mel_begin = d_mel * n as f32;
    let mel_center = d_mel * (n + 1) as f32;
    let mel_end = d_mel * (n + 2) as f32;

    let f_begin = to_hz(mel_begin, false);
    let f_center = to_hz(mel_center, false);
    let f_end = to_hz(mel_end, false);

    let i_begin = (f_begin / df).ceil() as usize;
    let i_center = (f_center / df).round_ties_even() as usize;
    let i_end = (f_end / df).floor() as usize;

    let mut sum = 0.0;
    for (i, spec_val) in spectrum
      .iter()
      .enumerate()
      .skip(i_begin + 1)
      .take(i_end - i_begin)
    {
      let f = df * i as f32;
      let mut a = if i < i_center {
        (f - f_begin) / (f_center - f_begin)
      } else {
        (f_end - f) / (f_end - f_center)
      };
      a /= (f_end - f_begin) * 0.5;
      sum += a * *spec_val;
    }
    *out_val = sum;
  }
}
