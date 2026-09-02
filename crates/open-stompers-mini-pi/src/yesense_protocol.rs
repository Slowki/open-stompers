//! Incremental parser for the Yesense IMU wire protocol.

use open_stompers_hardware::InertialMeasurement;
use open_stompers_hardware::InertialSnapshot;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::angular_velocity::degree_per_second;
use uom::si::f64::{Acceleration, AngularVelocity};

/// First sync byte in a Yesense frame.
pub const FRAME_HEADER_1: u8 = 0x59;
/// Second sync byte in a Yesense frame.
pub const FRAME_HEADER_2: u8 = 0x53;
/// Maximum payload size carried by a frame.
pub const MAX_PAYLOAD_LEN: usize = 0xFF;

/// Yesense TLV record identifier for linear acceleration.
pub const DATA_ID_ACCELERATION: u8 = 0x10;
/// Yesense TLV record identifier for angular velocity.
pub const DATA_ID_ANGULAR_VELOCITY: u8 = 0x20;
/// Yesense TLV record identifier for scalar-first orientation quaternions.
pub const DATA_ID_QUATERNION: u8 = 0x41;

const VECTOR_RECORD_SIZE: usize = 12;
const QUATERNION_RECORD_SIZE: usize = 16;
const FIXED_POINT_SCALE: f64 = 0.000_001;

/// Invalid frame conditions reported by the IMU parser.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum YesenseFrameError {
    /// The first checksum byte did not match.
    BadChecksumLow,
    /// The second checksum byte did not match.
    BadChecksumHigh,
    /// A data frame advertised an empty payload.
    EmptyPayload,
}

/// A complete Yesense frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedImuFrame {
    /// Transport identifier from the frame header.
    pub tid: u16,
    /// TLV records carried in the frame.
    pub payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReaderState {
    WaitHeader1,
    WaitHeader2,
    TidLow,
    TidHigh,
    Length,
    Payload,
    ChecksumLow,
    ChecksumHigh,
}

/// Stateful parser for fragmented Yesense frames.
#[derive(Debug)]
pub struct YesenseFrameReader {
    state: ReaderState,
    checksum_low: u8,
    checksum_high: u8,
    transport_id: u16,
    payload: Vec<u8>,
    payload_remaining: u8,
}

impl Default for YesenseFrameReader {
    fn default() -> Self {
        Self::new()
    }
}

impl YesenseFrameReader {
    /// Create a reader waiting for the frame sync bytes.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: ReaderState::WaitHeader1,
            checksum_low: 0,
            checksum_high: 0,
            transport_id: 0,
            payload: Vec::with_capacity(MAX_PAYLOAD_LEN),
            payload_remaining: 0,
        }
    }

    /// Feed a byte and return a complete frame when one is available.
    pub fn feed_byte(&mut self, byte: u8) -> Result<Option<DecodedImuFrame>, YesenseFrameError> {
        match self.state {
            ReaderState::WaitHeader1 => {
                if byte == FRAME_HEADER_1 {
                    self.state = ReaderState::WaitHeader2;
                }
            }
            ReaderState::WaitHeader2 => match byte {
                FRAME_HEADER_2 => {
                    self.checksum_low = 0;
                    self.checksum_high = 0;
                    self.payload.clear();
                    self.state = ReaderState::TidLow;
                }
                FRAME_HEADER_1 => {}
                _ => self.state = ReaderState::WaitHeader1,
            },
            ReaderState::TidLow => {
                self.accumulate(byte);
                self.transport_id = u16::from(byte);
                self.state = ReaderState::TidHigh;
            }
            ReaderState::TidHigh => {
                self.accumulate(byte);
                self.transport_id |= u16::from(byte) << 8;
                self.state = ReaderState::Length;
            }
            ReaderState::Length => {
                self.accumulate(byte);
                self.payload_remaining = byte;
                if byte == 0 {
                    self.reset();
                    return Err(YesenseFrameError::EmptyPayload);
                }
                self.state = ReaderState::Payload;
            }
            ReaderState::Payload => {
                self.accumulate(byte);
                self.payload.push(byte);
                self.payload_remaining -= 1;
                if self.payload_remaining == 0 {
                    self.state = ReaderState::ChecksumLow;
                }
            }
            ReaderState::ChecksumLow => {
                if byte != self.checksum_low {
                    self.reset();
                    return Err(YesenseFrameError::BadChecksumLow);
                }
                self.state = ReaderState::ChecksumHigh;
            }
            ReaderState::ChecksumHigh => {
                if byte != self.checksum_high {
                    self.reset();
                    return Err(YesenseFrameError::BadChecksumHigh);
                }
                let frame = DecodedImuFrame {
                    tid: self.transport_id,
                    payload: std::mem::take(&mut self.payload),
                };
                self.reset();
                return Ok(Some(frame));
            }
        }
        Ok(None)
    }

    fn accumulate(&mut self, byte: u8) {
        self.checksum_low = self.checksum_low.wrapping_add(byte);
        self.checksum_high = self.checksum_high.wrapping_add(self.checksum_low);
    }

    fn reset(&mut self) {
        self.state = ReaderState::WaitHeader1;
        self.checksum_low = 0;
        self.checksum_high = 0;
        self.transport_id = 0;
        self.payload.clear();
        self.payload_remaining = 0;
    }
}

/// Merge acceleration and angular velocity records from a frame into a sample.
pub fn merge_frame_payload(measurement: &mut InertialMeasurement, payload: &[u8]) {
    let mut cursor = 0;
    while cursor + 2 <= payload.len() {
        let identifier = payload[cursor];
        let data_length = usize::from(payload[cursor + 1]);
        let data_start = cursor + 2;
        let data_end = data_start + data_length;
        if data_end > payload.len() {
            cursor += 1;
            continue;
        }

        let data = &payload[data_start..data_end];
        if data_length == VECTOR_RECORD_SIZE && apply_vector_record(measurement, identifier, data) {
            cursor = data_end;
        } else {
            cursor += 1;
        }
    }
}

/// Merge valid records and refresh policy telemetry receive times.
pub fn merge_frame_snapshot(snapshot: &mut InertialSnapshot, payload: &[u8]) {
    let mut cursor = 0;
    while cursor + 2 <= payload.len() {
        let identifier = payload[cursor];
        let data_length = usize::from(payload[cursor + 1]);
        let data_start = cursor + 2;
        let data_end = data_start + data_length;
        if data_end > payload.len() {
            cursor += 1;
            continue;
        }

        let data = &payload[data_start..data_end];
        let now = std::time::Instant::now();
        match identifier {
            DATA_ID_ACCELERATION if data_length == VECTOR_RECORD_SIZE => {
                snapshot.measurement.acceleration =
                    decode_vector(data).map(Acceleration::new::<meter_per_second_squared>);
                cursor = data_end;
            }
            DATA_ID_ANGULAR_VELOCITY if data_length == VECTOR_RECORD_SIZE => {
                snapshot.measurement.angular_velocity =
                    decode_vector(data).map(AngularVelocity::new::<degree_per_second>);
                snapshot.angular_velocity_received_at = Some(now);
                cursor = data_end;
            }
            DATA_ID_QUATERNION if data_length == QUATERNION_RECORD_SIZE => {
                let quaternion = std::array::from_fn(|index| {
                    let offset = index * 4;
                    f64::from(i32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ])) * FIXED_POINT_SCALE
                });
                let norm = quaternion
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if norm.is_finite()
                    && norm > f64::EPSILON
                    && quaternion.iter().all(|value| value.is_finite())
                {
                    snapshot.orientation_wxyz = Some(quaternion.map(|value| value / norm));
                    snapshot.orientation_received_at = Some(now);
                }
                cursor = data_end;
            }
            _ => cursor += 1,
        }
    }
}

fn apply_vector_record(measurement: &mut InertialMeasurement, identifier: u8, data: &[u8]) -> bool {
    match identifier {
        DATA_ID_ACCELERATION => {
            measurement.acceleration =
                decode_vector(data).map(Acceleration::new::<meter_per_second_squared>);
            true
        }
        DATA_ID_ANGULAR_VELOCITY => {
            measurement.angular_velocity =
                decode_vector(data).map(AngularVelocity::new::<degree_per_second>);
            true
        }
        _ => false,
    }
}

fn decode_vector(data: &[u8]) -> [f64; 3] {
    std::array::from_fn(|index| {
        let offset = index * 4;
        f64::from(i32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])) * FIXED_POINT_SCALE
    })
}

#[cfg(test)]
pub(crate) fn encode_frame(transport_id: u16, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(7 + payload.len());
    frame.extend_from_slice(&[FRAME_HEADER_1, FRAME_HEADER_2]);
    frame.extend_from_slice(&transport_id.to_le_bytes());
    frame.push(payload.len() as u8);
    frame.extend_from_slice(payload);
    let mut checksum_low = 0u8;
    let mut checksum_high = 0u8;
    for byte in &frame[2..] {
        checksum_low = checksum_low.wrapping_add(*byte);
        checksum_high = checksum_high.wrapping_add(checksum_low);
    }
    frame.extend_from_slice(&[checksum_low, checksum_high]);
    frame
}

#[cfg(test)]
mod tests {
    use super::*;
    use uom::si::angular_velocity::radian_per_second;
    #[test]
    fn decodes_acceleration_and_angular_velocity_records() {
        let mut payload = vec![DATA_ID_ACCELERATION, VECTOR_RECORD_SIZE as u8];
        for value in [1_000_000_i32, -2_000_000, 500_000] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        payload.extend_from_slice(&[DATA_ID_ANGULAR_VELOCITY, VECTOR_RECORD_SIZE as u8]);
        for value in [1_000_000_i32, 0, -1_000_000] {
            payload.extend_from_slice(&value.to_le_bytes());
        }

        let mut reader = YesenseFrameReader::new();
        let frame = encode_frame(7, &payload)
            .into_iter()
            .find_map(|byte| reader.feed_byte(byte).expect("valid frame"))
            .expect("frame should decode");
        let mut measurement = InertialMeasurement {
            acceleration: [Acceleration::new::<meter_per_second_squared>(0.0); 3],
            angular_velocity: [AngularVelocity::new::<radian_per_second>(0.0); 3],
        };
        merge_frame_payload(&mut measurement, &frame.payload);

        assert_eq!(
            measurement.acceleration[0].get::<meter_per_second_squared>(),
            1.0
        );
        assert_eq!(
            measurement.acceleration[1].get::<meter_per_second_squared>(),
            -2.0
        );
        assert!(
            (measurement.angular_velocity[0].get::<radian_per_second>() - 1.0_f64.to_radians())
                .abs()
                < 1e-12
        );
        assert_eq!(
            measurement.angular_velocity[2].get::<radian_per_second>(),
            -1.0_f64.to_radians()
        );
        assert_eq!(
            measurement.acceleration[2].get::<meter_per_second_squared>(),
            0.5
        );
    }

    #[test]
    fn decodes_and_preserves_valid_orientation_records() {
        let mut snapshot = InertialSnapshot {
            measurement: InertialMeasurement {
                acceleration: [Acceleration::new::<meter_per_second_squared>(0.0); 3],
                angular_velocity: [AngularVelocity::new::<radian_per_second>(0.0); 3],
            },
            orientation_wxyz: None,
            angular_velocity_received_at: None,
            orientation_received_at: None,
        };
        let mut quaternion_payload = vec![DATA_ID_QUATERNION, QUATERNION_RECORD_SIZE as u8];
        for value in [1_000_000_i32, 0, 0, 0] {
            quaternion_payload.extend_from_slice(&value.to_le_bytes());
        }
        merge_frame_snapshot(&mut snapshot, &quaternion_payload);
        assert_eq!(snapshot.orientation_wxyz, Some([1.0, 0.0, 0.0, 0.0]));
        let orientation_received_at = snapshot.orientation_received_at;

        let mut invalid_quaternion_payload = vec![DATA_ID_QUATERNION, QUATERNION_RECORD_SIZE as u8];
        invalid_quaternion_payload.extend_from_slice(&[0; QUATERNION_RECORD_SIZE]);
        merge_frame_snapshot(&mut snapshot, &invalid_quaternion_payload);
        assert_eq!(snapshot.orientation_wxyz, Some([1.0, 0.0, 0.0, 0.0]));
        assert_eq!(snapshot.orientation_received_at, orientation_received_at);
    }
}
