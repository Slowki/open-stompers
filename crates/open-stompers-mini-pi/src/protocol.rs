//! High Torque motor CDC-serial frame format.
//!
//! Each frame is laid out as
//!
//! ```text
//! +------+-----+---------+------+---------+--- ... ---+
//! | 0xF7 | cmd | len (2) | crc8 | crc16(2)|  payload  |
//! +------+-----+---------+------+---------+--- ... ---+
//!   1B    1B     2B (LE)   1B     2B (LE)   len bytes
//! ```
//!
//! - `crc8` is [`crc::crc8`] applied to the 3 bytes `cmd | len_lo | len_hi`
//!   with init [`crc::CRC8_INIT`].
//! - `crc16` is [`crc::crc_ccitt`] applied to the `len` payload bytes with
//!   init [`crc::CRC16_INIT`].
//! - `len` is the payload byte count (max 256 per the reference SDK).
//!
//! The module exposes:
//! - [`CommandId`] for the subset of commands we support.
//! - [`encode_frame`] to serialize a `(cmd, payload)` pair.
//! - [`FrameReader`] which consumes a byte stream and yields complete frames.

use crate::crc::{CRC8_INIT, CRC16_INIT, crc_ccitt, crc8};

/// Leading sync byte of every frame.
pub const FRAME_HEAD: u8 = 0xF7;

/// Fixed frame header length: `head | cmd | len | crc8 | crc16`.
pub const HEADER_LEN: usize = 7;

/// Maximum payload byte count supported by the reference SDK.
pub const MAX_PAYLOAD_LEN: usize = 256;

/// Command identifiers exchanged on the wire.
///
/// This is a superset of the commands the controller sends (to allow decoding responses), but it is still a small subset of the reference SDK.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CommandId {
    /// Stop all motors (TX).
    Stop = 0x03,
    /// Set the number of motors on the CAN bus and query the comm board's
    /// firmware version (TX); RX carries `{..., major, minor}` in bytes 2-3.
    SetNum = 0x05,
    /// Motor state request/response using `{id, pos, val, tqe}` entries.
    MotorState = 0x06,
    /// Reset the motors (TX).
    Reset = 0x08,
    /// Motor state v2: `{id, mode, fault, pos, val, tqe}` entries.
    MotorState2 = 0x0A,
    /// Select the protocol feature version the comm board should speak (TX);
    /// RX echoes the selected version in byte 0.
    FunV = 0x0C,
    /// Reinitialize the comm board's FDCAN interface (TX). Required on boot
    /// for newer comm boards (`slave_version >= 4.1`) before the bus will
    /// forward messages to/from the motors.
    FdcanReset = 0x0E,
    /// Per-joint position command (TX).
    Position = 0x80,
    /// Motor command timeout (TX).
    Timeout = 0x85,
    /// Per-joint trajectory command: target position, max velocity, max
    /// torque (TX). Matches `motor::pos_vel_MAXtqe` in the reference SDK.
    PosVelMaxTorque = 0x90,
    /// Combined position / velocity / torque / Kp / Kd command (TX).
    ///
    /// Matches `motor::pos_vel_tqe_kp_kd` (`MODE_POS_VEL_TQE_KP_KD`) used by
    /// HighTorque midware position control. Payload is `{pos, vel, tqe, kp, kd}`
    /// per motor, each `i16`.
    PosVelTorqueKpKd = 0x93,
    /// Newer combined PD command (`MODE_POS_VEL_TQE_KP_KD2`). Same payload as
    /// [`Self::PosVelTorqueKpKd`]; only the command byte differs. Kept for
    /// decode completeness; Mini Pi Plus control uses [`Self::PosVelTorqueKpKd`].
    PosVelTorqueKpKd2 = 0xB0,
}

impl CommandId {
    /// Decode a raw command byte, returning `None` if it is unsupported.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x03 => Some(Self::Stop),
            0x06 => Some(Self::MotorState),
            0x05 => Some(Self::SetNum),
            0x0C => Some(Self::FunV),
            0x08 => Some(Self::Reset),
            0x0A => Some(Self::MotorState2),
            0x0E => Some(Self::FdcanReset),
            0x80 => Some(Self::Position),
            0x85 => Some(Self::Timeout),
            0x90 => Some(Self::PosVelMaxTorque),
            0x93 => Some(Self::PosVelTorqueKpKd),
            0xB0 => Some(Self::PosVelTorqueKpKd2),
            _ => None,
        }
    }
}

/// Append a complete frame for `cmd` and `payload` to `out`.
///
/// Returns an error when the payload exceeds [`MAX_PAYLOAD_LEN`].
pub fn encode_frame(cmd: CommandId, payload: &[u8], out: &mut Vec<u8>) -> Result<(), FrameError> {
    if payload.len() > MAX_PAYLOAD_LEN {
        return Err(FrameError::PayloadTooLarge(payload.len()));
    }

    let cmd_byte = cmd as u8;
    #[allow(clippy::cast_possible_truncation)]
    let len_lo = (payload.len() & 0xFF) as u8;
    #[allow(clippy::cast_possible_truncation)]
    let len_hi = ((payload.len() >> 8) & 0xFF) as u8;

    let header_check = [cmd_byte, len_lo, len_hi];
    let header_crc8 = crc8(&header_check, CRC8_INIT);
    let payload_crc16 = crc_ccitt(CRC16_INIT, payload);

    out.reserve(HEADER_LEN + payload.len());
    out.push(FRAME_HEAD);
    out.push(cmd_byte);
    out.push(len_lo);
    out.push(len_hi);
    out.push(header_crc8);
    out.extend_from_slice(&payload_crc16.to_le_bytes());
    out.extend_from_slice(payload);
    Ok(())
}

/// Errors produced while encoding or decoding frames.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameError {
    /// Payload exceeded [`MAX_PAYLOAD_LEN`].
    PayloadTooLarge(usize),
}

/// A single frame yielded by [`FrameReader`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedFrame {
    /// Decoded command identifier.
    pub command: CommandId,
    /// Payload bytes (exactly `len` long).
    pub payload: Vec<u8>,
}

/// Incremental frame parser that buffers partial data between reads.
///
/// Feed bytes from the serial port with [`FrameReader::extend`]; then call
/// [`FrameReader::next_frame`] in a loop until it returns `None`.
#[derive(Default)]
pub struct FrameReader {
    buffer: Vec<u8>,
}

impl FrameReader {
    /// Construct an empty reader.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append freshly read bytes to the internal buffer.
    pub fn extend(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Try to decode the next complete frame.
    ///
    /// # Returns
    ///
    /// - `Some(frame)` when a valid frame was parsed.
    /// - `None` when more bytes are needed.
    ///
    /// Unrecognized commands and CRC failures are silently skipped (matching
    /// the reference SDK), advancing past the bad byte so the next sync head
    /// byte can be found.
    pub fn next_frame(&mut self) -> Option<DecodedFrame> {
        loop {
            let Some(head_position) = self.buffer.iter().position(|&byte| byte == FRAME_HEAD)
            else {
                self.buffer.clear();
                return None;
            };
            if head_position > 0 {
                self.buffer.drain(..head_position);
            }

            if self.buffer.len() < HEADER_LEN {
                return None;
            }

            let cmd_byte = self.buffer[1];
            let len_lo = self.buffer[2];
            let len_hi = self.buffer[3];
            let header_crc = self.buffer[4];
            let payload_crc_lo = self.buffer[5];
            let payload_crc_hi = self.buffer[6];

            let header_bytes = [cmd_byte, len_lo, len_hi];
            if crc8(&header_bytes, CRC8_INIT) != header_crc {
                self.buffer.drain(..1);
                continue;
            }

            let payload_len = usize::from(u16::from_le_bytes([len_lo, len_hi]));
            if payload_len > MAX_PAYLOAD_LEN {
                self.buffer.drain(..1);
                continue;
            }

            let total_len = HEADER_LEN + payload_len;
            if self.buffer.len() < total_len {
                return None;
            }

            let payload = &self.buffer[HEADER_LEN..total_len];
            let expected_crc = u16::from_le_bytes([payload_crc_lo, payload_crc_hi]);
            if crc_ccitt(CRC16_INIT, payload) != expected_crc {
                self.buffer.drain(..1);
                continue;
            }

            let Some(command) = CommandId::from_u8(cmd_byte) else {
                self.buffer.drain(..total_len);
                continue;
            };

            let frame = DecodedFrame {
                command,
                payload: payload.to_vec(),
            };
            self.buffer.drain(..total_len);
            return Some(frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdcan_reset_request_matches_reference_wire_bytes() {
        // `canport::canboard_fdcan_reset` in the reference SDK sends
        // MODE_FDCAN_RESET with a 1-byte zero-filled payload.
        let mut wire = Vec::new();
        encode_frame(CommandId::FdcanReset, &[0x00], &mut wire).unwrap();
        assert_eq!(wire, vec![0xF7, 0x0E, 0x01, 0x00, 0xE2, 0x87, 0x0F, 0x00]);
    }

    #[test]
    fn motor_state2_request_matches_reference_wire_bytes() {
        // Encodes exactly what `canport::send_get_motor_state_cmd2` in the
        // reference SDK produces: head 0xF7, cmd 0x0A, len=1, CRC8 over
        // [cmd|len_lo|len_hi], CRC-CCITT over a one-byte payload of 0x7F.
        //
        // The expected bytes are the on-wire bytes computed from the exact
        // same CRC tables used by the reference firmware, so passing this
        // test means our transmit path is byte-for-byte identical.
        let mut wire = Vec::new();
        encode_frame(CommandId::MotorState2, &[0x7F], &mut wire).unwrap();
        assert_eq!(wire, vec![0xF7, 0x0A, 0x01, 0x00, 0x7C, 0xF7, 0x84, 0x7F]);
    }

    #[test]
    fn encode_then_decode_round_trips() {
        let payload = vec![0x01u8, 0x02, 0x03, 0x04, 0x05];
        let mut wire = Vec::new();
        encode_frame(CommandId::Position, &payload, &mut wire).unwrap();

        let mut reader = FrameReader::new();
        reader.extend(&wire);
        let frame = reader.next_frame().unwrap();
        assert_eq!(frame.command, CommandId::Position);
        assert_eq!(frame.payload, payload);
        assert!(reader.next_frame().is_none());
    }

    #[test]
    fn decoder_skips_garbage_before_head() {
        let payload = vec![0xAA, 0xBB];
        let mut wire = vec![0x00, 0x12, 0x34];
        encode_frame(CommandId::Reset, &payload, &mut wire).unwrap();
        wire.extend_from_slice(&[0x99, 0x98]);

        let mut reader = FrameReader::new();
        reader.extend(&wire);
        let frame = reader.next_frame().unwrap();
        assert_eq!(frame.command, CommandId::Reset);
        assert_eq!(frame.payload, payload);
    }

    #[test]
    fn decoder_recovers_after_bad_crc8() {
        let mut wire = Vec::new();
        encode_frame(CommandId::Stop, &[0x7F], &mut wire).unwrap();
        wire[4] ^= 0x01;
        let mut good = Vec::new();
        encode_frame(CommandId::Stop, &[0x7F], &mut good).unwrap();
        wire.extend_from_slice(&good);

        let mut reader = FrameReader::new();
        reader.extend(&wire);
        let frame = reader.next_frame().unwrap();
        assert_eq!(frame.command, CommandId::Stop);
        assert_eq!(frame.payload, vec![0x7F]);
    }

    #[test]
    fn decoder_drops_unknown_commands() {
        let payload = [0x00u8; 4];
        let len_lo = payload.len() as u8;
        let header_bytes = [0x77, len_lo, 0x00];
        let mut wire = Vec::from([FRAME_HEAD, 0x77, len_lo, 0x00]);
        wire.push(crc8(&header_bytes, CRC8_INIT));
        wire.extend_from_slice(&crc_ccitt(CRC16_INIT, &payload).to_le_bytes());
        wire.extend_from_slice(&payload);
        encode_frame(CommandId::Stop, &[0x7F], &mut wire).unwrap();

        let mut reader = FrameReader::new();
        reader.extend(&wire);
        let frame = reader.next_frame().unwrap();
        assert_eq!(frame.command, CommandId::Stop);
    }

    #[test]
    fn decoder_waits_for_partial_payload() {
        let payload = vec![1, 2, 3, 4];
        let mut wire = Vec::new();
        encode_frame(CommandId::Position, &payload, &mut wire).unwrap();

        let mut reader = FrameReader::new();
        reader.extend(&wire[..wire.len() - 2]);
        assert!(reader.next_frame().is_none());
        reader.extend(&wire[wire.len() - 2..]);
        let frame = reader.next_frame().unwrap();
        assert_eq!(frame.payload, payload);
    }

    #[test]
    fn pos_vel_torque_kp_kd_uses_midware_command_byte() {
        assert_eq!(CommandId::PosVelTorqueKpKd as u8, 0x93);
        assert_eq!(CommandId::from_u8(0x93), Some(CommandId::PosVelTorqueKpKd));

        let mut wire = Vec::new();
        encode_frame(CommandId::PosVelTorqueKpKd, &[0x00], &mut wire).unwrap();
        assert_eq!(wire[1], 0x93);
    }
}

#[cfg(test)]
mod buffering_tests {
    use super::*;

    #[test]
    fn garbage_and_incomplete_headers_use_bounded_storage() {
        let mut reader = FrameReader::new();
        for _ in 0..100 {
            reader.extend(&[0x00; 512]);
            assert!(reader.next_frame().is_none());
            assert!(reader.buffer.is_empty());
        }
        reader.extend(&[0x00, 0x00, FRAME_HEAD, 0x0A]);
        assert!(reader.next_frame().is_none());
        assert_eq!(reader.buffer, [FRAME_HEAD, 0x0A]);
    }
}
