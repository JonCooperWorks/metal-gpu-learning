// =============================================================================
// LESSON 9 TOKENIZER: minimal byte-level tokenizer used for training/inference
// =============================================================================
// Why this tokenizer exists:
// - Keep the lesson transparent and dependency-light.
// - Make it obvious how text maps to model IDs.
//
// Vocabulary layout:
// - IDs 0..=255: raw UTF-8 bytes
// - 256: <pad>
// - 257: <bos> (beginning of sequence)
// - 258: <eos> (end of sequence)
//
// This is not a production tokenizer. It is intentionally simple so you can
// focus on transformer math and GPU execution.
// =============================================================================

pub const PAD_ID: u32 = 256;
pub const BOS_ID: u32 = 257;
pub const EOS_ID: u32 = 258;
pub const VOCAB_SIZE: usize = 259;

/// Encodes text to IDs with explicit BOS/EOS markers.
///
/// Mapping is straightforward:
///   id = byte_value for every byte in the string.
/// Example:
///   "Hi" bytes [72, 105] -> [BOS, 72, 105, EOS]
pub fn encode_with_specials(text: &str) -> Vec<u32> {
    let mut out = Vec::with_capacity(text.len() + 2);
    out.push(BOS_ID);
    out.extend(text.as_bytes().iter().map(|b| *b as u32));
    out.push(EOS_ID);
    out
}

/// Decodes IDs back into text bytes.
///
/// Non-byte IDs (special tokens) are skipped by design.
pub fn decode_ids(ids: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(ids.len());
    for &id in ids {
        if id <= 255 {
            bytes.push(id as u8);
        }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_round_trip() {
        let text = "Hello GPU";
        let ids = encode_with_specials(text);
        assert_eq!(ids[0], BOS_ID);
        assert_eq!(ids[ids.len() - 1], EOS_ID);
        let decoded = decode_ids(&ids);
        assert_eq!(decoded, text);
    }
}
