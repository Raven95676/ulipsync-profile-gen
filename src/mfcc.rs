use crate::algorithm;

pub fn extract_mfcc(
  mut input: Vec<f32>,
  input_sample_rate: u32,
  target_sample_rate: u32,
  mel_filter_bank_channels: usize,
) -> Vec<f32> {
  const RANGE: f32 = 500.0;
  let cutoff = target_sample_rate as f32 / 2.0;

  // 低通 + 降采样 + 预加重 + 汉明窗 + 归一化
  algorithm::low_pass_filter(&mut input, input_sample_rate as f32, cutoff, RANGE);
  let mut data = algorithm::downsample(&input, input_sample_rate, target_sample_rate);
  algorithm::pre_emphasis(&mut data, 0.97);
  algorithm::hamming(&mut data);
  algorithm::normalize(&mut data, 1.0);

  // 频谱 -> Mel滤波 -> dB -> DCT -> MFCC（跳过第0项）
  let spectrum = algorithm::fft(&data);
  let mut mel_spectrum = algorithm::mel_filter_bank(
    &spectrum,
    target_sample_rate as f32,
    mel_filter_bank_channels,
  );
  algorithm::power_to_db(&mut mel_spectrum);
  let mel_cepstrum = algorithm::dct(&mel_spectrum);

  let mfcc: Vec<f32> = mel_cepstrum.iter().skip(1).copied().collect();

  mfcc
}
