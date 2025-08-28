#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod algorithm;
mod mfcc;

const MFCC_SIZE: usize = 12;

#[derive(Serialize, Deserialize, Clone)]
struct MfccCalibrationData {
  array: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone)]
struct MfccEntry {
  name: String,
  #[serde(rename = "mfccCalibrationDataList")]
  mfcc_calibration_data_list: Vec<MfccCalibrationData>,
}

#[derive(Serialize, Deserialize)]
struct OutputJson {
  #[serde(rename = "mfccNum")]
  mfcc_num: usize,
  #[serde(rename = "mfccDataCount")]
  mfcc_data_count: usize,
  #[serde(rename = "melFilterBankChannels")]
  mel_filter_bank_channels: usize,
  #[serde(rename = "targetSampleRate")]
  target_sample_rate: u32,
  #[serde(rename = "sampleCount")]
  sample_count: usize,
  #[serde(rename = "useStandardization")]
  use_standardization: u32,
  #[serde(rename = "compareMethod")]
  compare_method: u32,
  #[serde(rename = "mfccs")]
  mfccs: Vec<MfccEntry>,
}

#[napi]
pub enum CompareMethod {
  L1Norm,
  L2Norm,
  CosineSimilarity,
}

impl CompareMethod {
  fn as_u32(&self) -> u32 {
    match self {
      CompareMethod::L1Norm => 0,
      CompareMethod::L2Norm => 1,
      CompareMethod::CosineSimilarity => 2,
    }
  }
}

#[napi]
pub struct ProfileGenerator {
  target_sample_rate: u32,
  mel_filter_bank_channels: usize,
  compare_method: CompareMethod,
  entries: HashMap<String, Vec<MfccCalibrationData>>,
  mfcc_data_count: usize,
  sample_count: usize,
  use_standardization: bool,
}

#[napi(object)]
pub struct ProfileGeneratorOptions {
  pub target_sample_rate: u32,
  pub mel_filter_bank_channels: u32,
  pub compare_method: Option<CompareMethod>,
  pub mfcc_data_count: Option<u32>,
  pub sample_count: Option<u32>,
  pub use_standardization: Option<bool>,
}

#[napi]
impl ProfileGenerator {
  #[napi(constructor)]
  pub fn new(opts: ProfileGeneratorOptions) -> Self {
    Self {
      target_sample_rate: opts.target_sample_rate,
      mel_filter_bank_channels: opts.mel_filter_bank_channels as usize,
      compare_method: opts.compare_method.unwrap_or(CompareMethod::L2Norm),
      entries: HashMap::new(),
      mfcc_data_count: opts.mfcc_data_count.unwrap_or(16) as usize,
      sample_count: opts.sample_count.unwrap_or(1024) as usize,
      use_standardization: opts.use_standardization.unwrap_or(false),
    }
  }

  #[napi]
  pub fn add_sample(
    &mut self,
    audio: Float32Array,
    phoneme_name: String,
    input_sample_rate: u32,
  ) -> Result<()> {
    if audio.is_empty() {
      return Err(Error::new(Status::InvalidArg, "Audio data is empty"));
    }

    let audio_data: Vec<f32> = audio.to_vec();
    let total = audio_data.len();

    thread_local! {
      static MFCC_POOL: std::cell::RefCell<mfcc::MfccBufferPool> = std::cell::RefCell::new(mfcc::MfccBufferPool::new());
    }

    let mut mfcc_output: Vec<f32> = Vec::new();
    let mut frame_buf: Vec<f32> = vec![0.0; self.sample_count];

    let entry_list = self.entries.entry(phoneme_name).or_default();

    let mut start = 0usize;
    while start + self.sample_count <= total {
      let end = start + self.sample_count;
      frame_buf.copy_from_slice(&audio_data[start..end]);
      MFCC_POOL.with(|pool_ref| {
        let mut pool = pool_ref.borrow_mut();
        mfcc::extract_mfcc(
          &mut frame_buf,
          input_sample_rate,
          self.target_sample_rate,
          self.mel_filter_bank_channels,
          &mut pool,
          &mut mfcc_output,
        );
      });

      if mfcc_output.iter().any(|&v| !v.is_finite()) {
        start += self.sample_count;
        continue;
      }
      let result_data = std::mem::take(&mut mfcc_output);
      let calibration_data = MfccCalibrationData { array: result_data };
      entry_list.push(calibration_data);
      if entry_list.len() > self.mfcc_data_count {
        let overflow = entry_list.len() - self.mfcc_data_count;
        entry_list.drain(0..overflow);
      }

      start += self.sample_count;
    }

    Ok(())
  }

  #[napi]
  pub fn finish(&mut self) -> Result<String> {
    let mfcc_entries: Vec<MfccEntry> = self
      .entries
      .drain()
      .map(|(name, data_list)| MfccEntry {
        name,
        mfcc_calibration_data_list: data_list,
      })
      .collect();

    let output = OutputJson {
      mfcc_num: MFCC_SIZE,
      mfcc_data_count: self.mfcc_data_count,
      mel_filter_bank_channels: self.mel_filter_bank_channels,
      target_sample_rate: self.target_sample_rate,
      sample_count: self.sample_count,
      use_standardization: if self.use_standardization { 1 } else { 0 },
      compare_method: self.compare_method.as_u32(),
      mfccs: mfcc_entries,
    };

    serde_json::to_string_pretty(&output)
      .map_err(|e| Error::new(Status::GenericFailure, format!("Serialization error: {e}")))
  }
}
