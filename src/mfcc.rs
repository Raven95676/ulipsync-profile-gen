use crate::algorithm;
use crate::MFCC_SIZE;
use rustfft::num_complex::Complex32;

pub struct MfccBufferPool {
  downsample: Vec<f32>,
  fft_complex: Vec<Complex32>,
  spectrum: Vec<f32>,
  mel_spectrum: Vec<f32>,
  cepstrum: Vec<f32>,
}

impl MfccBufferPool {
  pub fn new() -> Self {
    Self {
      downsample: Vec::new(),
      fft_complex: Vec::new(),
      spectrum: Vec::new(),
      mel_spectrum: Vec::new(),
      cepstrum: Vec::new(),
    }
  }
}

pub fn extract_mfcc(
  input: &mut [f32],
  input_sample_rate: u32,
  target_sample_rate: u32,
  mel_filter_bank_channels: usize,
  pool: &mut MfccBufferPool,
  out: &mut Vec<f32>,
) {
  const RANGE: f32 = 500.0;
  let cutoff = target_sample_rate as f32 / 2.0;

  // 低通 + 降采样 + 预加重 + 汉明窗 + 归一化
  algorithm::low_pass_filter(input, input_sample_rate as f32, cutoff, RANGE);
  algorithm::downsample(
    input,
    input_sample_rate,
    target_sample_rate,
    &mut pool.downsample,
  );
  algorithm::pre_emphasis(&mut pool.downsample, 0.97);
  algorithm::hamming(&mut pool.downsample);
  algorithm::normalize(&mut pool.downsample, 1.0);

  // 频谱 -> Mel滤波 -> dB -> DCT -> MFCC（跳过第0项）
  algorithm::fft(&pool.downsample, &mut pool.fft_complex, &mut pool.spectrum);
  if pool.mel_spectrum.len() != mel_filter_bank_channels {
    pool.mel_spectrum.resize(mel_filter_bank_channels, 0.0);
  }
  algorithm::mel_filter_bank(
    &pool.spectrum,
    target_sample_rate as f32,
    mel_filter_bank_channels,
    &mut pool.mel_spectrum,
  );
  algorithm::power_to_db(&mut pool.mel_spectrum);
  if pool.cepstrum.len() != mel_filter_bank_channels {
    pool.cepstrum.resize(mel_filter_bank_channels, 0.0);
  }
  algorithm::dct(&pool.mel_spectrum, &mut pool.cepstrum);
  out.clear();
  out.reserve(MFCC_SIZE);
  out.extend(pool.cepstrum.iter().skip(1).take(MFCC_SIZE));
}
