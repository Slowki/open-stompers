//! Dynamixel protocol 2.0 packets used by the Microduck bus.
//!
//! Instruction packets are byte-stuffed. Fast sync read status packets are not:
//! each device appends `[error, id, data, running-crc]` to one broadcast status
//! packet, and that running CRC is checked per block.

use thiserror::Error;

/// Broadcast ID used by sync read and sync write.
pub const BROADCAST_ID: u8 = 0xFE;

/// Protocol 2.0 sync read instruction.
pub const INSTRUCTION_SYNC_READ: u8 = 0x82;
/// Protocol 2.0 sync write instruction.
pub const INSTRUCTION_SYNC_WRITE: u8 = 0x83;
/// Protocol 2.0 fast sync read instruction.
pub const INSTRUCTION_FAST_SYNC_READ: u8 = 0x8A;
/// Status-packet instruction byte.
pub const INSTRUCTION_STATUS: u8 = 0x55;

/// `present_pwm`, the start of the 12-byte block shared with the IMU board.
pub const PRESENT_BLOCK_ADDR: u16 = 124;
/// Bytes in the combined present-state and IMU block.
pub const PRESENT_BLOCK_LEN: u16 = 12;
/// `goal_position` register.
pub const GOAL_POSITION_ADDR: u16 = 116;
/// Width of `goal_position`.
pub const GOAL_POSITION_LEN: u16 = 4;
/// `torque_enable` register.
pub const TORQUE_ENABLE_ADDR: u16 = 64;
/// `position_d_gain`, followed by I and P gains.
pub const POSITION_D_GAIN_ADDR: u16 = 80;
/// Combined width of the D, I, and P gain registers.
pub const POSITION_GAIN_LEN: u16 = 6;
/// Maximum XL330 position gain register value.
pub const POSITION_GAIN_MAX: u16 = 16_383;

const HEADER_LEN: usize = 7;
const MAX_BODY_LEN: usize = 1024;

/// A CRC-validated protocol 2.0 packet, still in on-wire form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawPacket {
    /// Packet identifier.
    pub id: u8,
    /// Instruction byte.
    pub instruction: u8,
    /// Complete packet, including the header and CRC.
    pub bytes: Vec<u8>,
}

/// Failures while interpreting a CRC-valid packet.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ProtocolError {
    /// The packet ended before the expected fields.
    #[error("truncated dynamixel packet")]
    Truncated,
    /// A fast sync read status packet did not match the requested shape.
    #[error("fast sync read status length or identifier mismatch")]
    FastSyncReadMismatch,
    /// A status packet did not answer the requested device.
    #[error("status packet id {actual} did not match requested id {expected}")]
    UnexpectedId {
        /// Identifier the host asked for.
        expected: u8,
        /// Identifier in the status packet.
        actual: u8,
    },
}

/// Incremental protocol 2.0 packet scanner.
#[derive(Debug, Default)]
pub struct PacketReader {
    buffer: Vec<u8>,
}

impl PacketReader {
    /// Create an empty scanner.
    #[must_use]
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Discard buffered bytes after a timed-out transaction.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Append newly received bytes.
    pub fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Return the next CRC-valid packet, skipping bytes that cannot start one.
    pub fn next_packet(&mut self) -> Option<RawPacket> {
        loop {
            let Some(start) = find_header(&self.buffer) else {
                if self.buffer.len() > 3 {
                    let keep_from = self.buffer.len() - 3;
                    self.buffer.drain(..keep_from);
                }
                return None;
            };
            if start > 0 {
                self.buffer.drain(..start);
            }
            if self.buffer.len() < HEADER_LEN {
                return None;
            }
            let body_len = u16::from_le_bytes([self.buffer[5], self.buffer[6]]) as usize;
            if !(3..=MAX_BODY_LEN).contains(&body_len) {
                self.buffer.drain(..1);
                continue;
            }
            let total = HEADER_LEN + body_len;
            if self.buffer.len() < total {
                return None;
            }
            let computed = crc(&self.buffer[..total - 2]);
            let received = u16::from_le_bytes([self.buffer[total - 2], self.buffer[total - 1]]);
            if computed != received {
                self.buffer.drain(..1);
                continue;
            }
            let bytes = self.buffer.drain(..total).collect::<Vec<_>>();
            return Some(RawPacket {
                id: bytes[4],
                instruction: bytes[7],
                bytes,
            });
        }
    }
}

/// Encode an instruction packet, stuffing `0xFF 0xFF 0xFD` in the parameters.
#[must_use]
pub fn encode_instruction(id: u8, instruction: u8, params: &[u8]) -> Vec<u8> {
    let stuffed = add_stuffing(params);
    let length = u16::try_from(stuffed.len() + 3).expect("instruction packet fits in u16");
    let mut bytes = Vec::with_capacity(HEADER_LEN + usize::from(length));
    bytes.extend_from_slice(&[0xFF, 0xFF, 0xFD, 0x00, id]);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.push(instruction);
    bytes.extend_from_slice(&stuffed);
    let checksum = crc(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    bytes
}

/// Parameters of a sync or fast sync read.
#[must_use]
pub fn sync_read_params(address: u16, length: u16, ids: &[u8]) -> Vec<u8> {
    let mut params = Vec::with_capacity(4 + ids.len());
    params.extend_from_slice(&address.to_le_bytes());
    params.extend_from_slice(&length.to_le_bytes());
    params.extend_from_slice(ids);
    params
}

/// Parameters of a sync write. Every value must have the same length.
#[must_use]
pub fn sync_write_params(address: u16, values: &[(u8, &[u8])]) -> Vec<u8> {
    let data_len = values.first().map_or(0, |(_, data)| data.len());
    let mut params = Vec::with_capacity(4 + values.len() * (1 + data_len));
    params.extend_from_slice(&address.to_le_bytes());
    params.extend_from_slice(
        &u16::try_from(data_len)
            .expect("sync write field fits in u16")
            .to_le_bytes(),
    );
    for (id, data) in values {
        debug_assert_eq!(data.len(), data_len);
        params.push(*id);
        params.extend_from_slice(data);
    }
    params
}

/// Instruction parameters with byte stuffing removed.
#[cfg(test)]
#[must_use]
pub fn instruction_params(packet: &RawPacket) -> Vec<u8> {
    remove_stuffing(&packet.bytes[HEADER_LEN + 1..packet.bytes.len() - 2])
}

/// Error byte and parameters of a normal status packet.
///
/// # Errors
///
/// Returns [`ProtocolError::Truncated`] when the destuffed body has no error byte.
pub fn status_body(packet: &RawPacket) -> Result<(u8, Vec<u8>), ProtocolError> {
    let stuffed = &packet.bytes[HEADER_LEN + 1..packet.bytes.len() - 2];
    let body = remove_stuffing(stuffed);
    let Some((error, params)) = body.split_first() else {
        return Err(ProtocolError::Truncated);
    };
    Ok((*error, params.to_vec()))
}

/// Split a fast sync read status packet into one data block per requested ID.
///
/// # Errors
///
/// Returns [`ProtocolError::FastSyncReadMismatch`] when the packet shape or a
/// block CRC does not match the request.
pub fn parse_fast_sync_read(
    packet: &RawPacket,
    ids: &[u8],
    length: u16,
) -> Result<Vec<Vec<u8>>, ProtocolError> {
    let data_len = usize::from(length);
    let block_len = data_len + 4;
    let body_start = HEADER_LEN + 1;
    if packet.id != BROADCAST_ID
        || packet.instruction != INSTRUCTION_STATUS
        || packet.bytes.len() != body_start + ids.len() * block_len
    {
        return Err(ProtocolError::FastSyncReadMismatch);
    }
    let mut values = Vec::with_capacity(ids.len());
    for (index, id) in ids.iter().enumerate() {
        let block = body_start + index * block_len;
        let crc_at = block + 2 + data_len;
        let read_crc = u16::from_le_bytes([packet.bytes[crc_at], packet.bytes[crc_at + 1]]);
        if read_crc != crc(&packet.bytes[..crc_at]) || packet.bytes[block + 1] != *id {
            return Err(ProtocolError::FastSyncReadMismatch);
        }
        values.push(packet.bytes[block + 2..crc_at].to_vec());
    }
    Ok(values)
}

/// Build a fast sync read status packet. `blocks` are `(id, data)` in reply order.
#[cfg(test)]
#[must_use]
pub fn encode_fast_sync_status(blocks: &[(u8, &[u8])]) -> Vec<u8> {
    let data_len = blocks.first().map_or(0, |(_, data)| data.len());
    let body_len = 1 + blocks.len() * (data_len + 4);
    let mut bytes = Vec::with_capacity(HEADER_LEN + body_len);
    bytes.extend_from_slice(&[0xFF, 0xFF, 0xFD, 0x00, BROADCAST_ID]);
    bytes.extend_from_slice(
        &u16::try_from(body_len)
            .expect("status packet fits in u16")
            .to_le_bytes(),
    );
    bytes.push(INSTRUCTION_STATUS);
    for (id, data) in blocks {
        debug_assert_eq!(data.len(), data_len);
        bytes.push(0);
        bytes.push(*id);
        bytes.extend_from_slice(data);
        let checksum = crc(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
    }
    bytes
}

fn find_header(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(4)
        .position(|window| window == [0xFF, 0xFF, 0xFD, 0x00])
}

fn add_stuffing(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut run = 0_u8;
    for &byte in data {
        out.push(byte);
        run = match (run, byte) {
            (0, 0xFF) | (1, 0xFF) => run + 1,
            (2, 0xFF) => 2,
            (2, 0xFD) => {
                out.push(0xFD);
                0
            }
            (_, 0xFF) => 1,
            _ => 0,
        };
    }
    out
}

fn remove_stuffing(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut run = 0_u8;
    for &byte in data {
        if run == 3 {
            run = 0;
            if byte == 0xFD {
                continue;
            }
        }
        run = match (run, byte) {
            (0, 0xFF) | (1, 0xFF) => run + 1,
            (2, 0xFF) => 2,
            (2, 0xFD) => 3,
            (_, 0xFF) => 1,
            _ => 0,
        };
        out.push(byte);
    }
    out
}

fn crc(data: &[u8]) -> u16 {
    let mut crc_accum = 0_u16;
    for byte in data {
        let index = usize::from((crc_accum >> 8) as u8 ^ byte);
        crc_accum = (crc_accum << 8) ^ CRC_TABLE[index];
    }
    crc_accum
}

const CRC_TABLE: [u16; 256] = [
    0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011, 0x8033, 0x0036, 0x003C, 0x8039,
    0x0028, 0x802D, 0x8027, 0x0022, 0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072,
    0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041, 0x80C3, 0x00C6, 0x00CC, 0x80C9,
    0x00D8, 0x80DD, 0x80D7, 0x00D2, 0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1,
    0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1, 0x8093, 0x0096, 0x009C, 0x8099,
    0x0088, 0x808D, 0x8087, 0x0082, 0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192,
    0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1, 0x01E0, 0x81E5, 0x81EF, 0x01EA,
    0x81FB, 0x01FE, 0x01F4, 0x81F1, 0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2,
    0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151, 0x8173, 0x0176, 0x017C, 0x8179,
    0x0168, 0x816D, 0x8167, 0x0162, 0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
    0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101, 0x8303, 0x0306, 0x030C, 0x8309,
    0x0318, 0x831D, 0x8317, 0x0312, 0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321,
    0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371, 0x8353, 0x0356, 0x035C, 0x8359,
    0x0348, 0x834D, 0x8347, 0x0342, 0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1,
    0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2, 0x83A3, 0x03A6, 0x03AC, 0x83A9,
    0x03B8, 0x83BD, 0x83B7, 0x03B2, 0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381,
    0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291, 0x82B3, 0x02B6, 0x02BC, 0x82B9,
    0x02A8, 0x82AD, 0x82A7, 0x02A2, 0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2,
    0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1, 0x8243, 0x0246, 0x024C, 0x8249,
    0x0258, 0x825D, 0x8257, 0x0252, 0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261,
    0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231, 0x8213, 0x0216, 0x021C, 0x8219,
    0x0208, 0x820D, 0x8207, 0x0202,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_the_protocol_vector() {
        let data = [0xFF, 0xFF, 0xFD, 0x00, 0x2A, 0x03, 0x00, 0x01];
        assert_eq!(crc(&data).to_le_bytes(), [0x16, 0xD2]);
    }

    #[test]
    fn fast_sync_read_instruction_matches_the_reference_layout() {
        let params = sync_read_params(132, 4, &[1, 2]);
        let bytes = encode_instruction(BROADCAST_ID, INSTRUCTION_FAST_SYNC_READ, &params);
        assert_eq!(bytes[7], INSTRUCTION_FAST_SYNC_READ);
        assert_eq!(bytes[..7], [0xFF, 0xFF, 0xFD, 0x00, 0xFE, 0x09, 0x00]);
        assert_eq!(bytes[8..12], [0x84, 0x00, 0x04, 0x00]);
        assert_eq!(bytes[12..14], [0x01, 0x02]);
        assert_eq!(crc(&bytes[..bytes.len() - 2]).to_le_bytes(), bytes[14..]);
    }

    #[test]
    fn write_parameters_containing_a_header_are_stuffed() {
        let bytes = encode_instruction(1, 0x03, &[0x74, 0x00, 0xFF, 0xFF, 0xFD, 0x00]);
        assert_eq!(bytes[5..7], [0x0A, 0x00]);
        assert_eq!(bytes[8..15], [0x74, 0x00, 0xFF, 0xFF, 0xFD, 0xFD, 0x00]);
    }

    #[test]
    fn status_parameters_are_destuffed() {
        let mut bytes = vec![
            0xFF, 0xFF, 0xFD, 0x00, 0x01, 0x09, 0x00, 0x55, 0x00, 0xFF, 0xFF, 0xFD, 0xFD, 0xA6,
        ];
        bytes.extend_from_slice(&crc(&bytes).to_le_bytes());
        let mut reader = PacketReader::new();
        reader.push(&bytes);
        let packet = reader.next_packet().expect("status packet");
        let (error, params) = status_body(&packet).expect("status body");
        assert_eq!(error, 0);
        assert_eq!(params, [0xFF, 0xFF, 0xFD, 0xA6]);
    }

    #[test]
    fn fast_sync_status_matches_the_emanual_example() {
        const EXAMPLE: [u8; 32] = [
            0xFF, 0xFF, 0xFD, 0x00, 0xFE, 0x19, 0x00, 0x55, 0x00, 0x03, 0xA6, 0x00, 0x00, 0x00,
            0x84, 0x08, 0x00, 0x07, 0x1F, 0x08, 0x00, 0x00, 0x16, 0xCA, 0x00, 0x04, 0xFF, 0x03,
            0x00, 0x00, 0xD1, 0x9E,
        ];
        let blocks: [&[u8]; 3] = [
            &[0xA6, 0x00, 0x00, 0x00],
            &[0x1F, 0x08, 0x00, 0x00],
            &[0xFF, 0x03, 0x00, 0x00],
        ];
        let encoded = encode_fast_sync_status(&[(3, blocks[0]), (7, blocks[1]), (4, blocks[2])]);
        assert_eq!(encoded, EXAMPLE);
        let mut reader = PacketReader::new();
        reader.push(&EXAMPLE);
        let packet = reader.next_packet().expect("example packet");
        let values = parse_fast_sync_read(&packet, &[3, 7, 4], 4).expect("parsed blocks");
        assert_eq!(values[0], blocks[0]);
        assert_eq!(values[1], blocks[1]);
        assert_eq!(values[2], blocks[2]);
    }

    #[test]
    fn corrupt_crc_is_skipped_until_a_valid_packet() {
        let valid = encode_instruction(1, 0x01, &[]);
        let mut bytes = vec![0xFF, 0xFF, 0xFD, 0x00, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00];
        bytes.extend_from_slice(&valid);
        let mut reader = PacketReader::new();
        reader.push(&bytes);
        let packet = reader.next_packet().expect("valid packet after garbage");
        assert_eq!(packet.bytes, valid);
    }
}
