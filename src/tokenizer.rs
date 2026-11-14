//! Tokenizers loaded from GGUF metadata.
//!
//! Two families are supported:
//!
//! * `gpt2`: byte-level BPE with a family-specific pre-tokenizer regex
//!   (GPT-2, SmolLM, Llama 3, Qwen 2/3, ...).
//! * `llama`: SentencePiece-style BPE over scores with byte fallback
//!   (Llama 1/2, Mistral, TinyLlama, ...).
//!
//! Special (control and user-defined) tokens are matched literally before
//! the ordinary algorithm runs, the way llama.cpp does it.

use std::{cmp::Ordering, collections::BinaryHeap, collections::HashMap, fmt};

use fancy_regex::Regex;

use crate::gguf::{GgufInfo, Value};

pub type TokenId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_code(code: i64) -> Self {
        match code {
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

#[derive(Debug)]
pub struct TokenizerError(pub String);

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tokenizer: {}", self.0)
    }
}

impl std::error::Error for TokenizerError {}

fn err(message: impl Into<String>) -> TokenizerError {
    TokenizerError(message.into())
}

// One tokenizer exists per model; the size difference between variants is moot.
#[allow(clippy::large_enum_variant)]
enum Algorithm {
    Bpe {
        /// (left, right) -> (rank, merged id)
        merges: HashMap<(TokenId, TokenId), (u32, TokenId)>,
        pre: Vec<Regex>,
        /// Emit a whole pre-token directly when it is itself in the vocabulary.
        ignore_merges: bool,
        byte_to_char: [char; 256],
        char_to_byte: HashMap<char, u8>,
        byte_tokens: [Option<TokenId>; 256],
    },
    Spm {
        scores: Vec<f32>,
        add_space_prefix: bool,
        byte_tokens: [Option<TokenId>; 256],
    },
}

pub struct Tokenizer {
    tokens: Vec<String>,
    types: Vec<TokenType>,
    lookup: HashMap<String, TokenId>,
    /// Literal special tokens, longest first.
    specials: Vec<(String, TokenId, TokenType)>,
    algorithm: Algorithm,
    pub bos: Option<TokenId>,
    pub eos: Option<TokenId>,
    /// End-of-generation tokens (EOS, EOT, `<|im_end|>`, ...).
    pub eog: Vec<TokenId>,
    pub add_bos: bool,
    pub model_name: String,
    pub pre_name: String,
}

// Pre-tokenizer patterns, transcribed from llama.cpp's llama-vocab.cpp.
const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const DIGITS: &str = r"\p{N}";

fn pre_patterns(pre: &str) -> Result<(Vec<&'static str>, bool), TokenizerError> {
    // (patterns applied in sequence, ignore_merges)
    Ok(match pre {
        "default" | "gpt2" | "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "starcoder" | "refact"
        | "olmo" | "mpt" => (vec![GPT2], false),
        "smollm" | "codeshell" | "exaone" | "jais" => (vec![DIGITS, GPT2], false),
        "llama3" | "llama-bpe" | "llama-v3" | "smaug-bpe" | "falcon3" | "pixtral" | "dbrx"
        | "superbpe" => (vec![LLAMA3], true),
        "qwen2" | "qwen35" | "deepseek-r1-qwen" | "stablelm2" | "megrez" | "hunyuan" => {
            (vec![QWEN2], false)
        }
        other => return Err(err(format!("unsupported pre-tokenizer {other:?}"))),
    })
}

/// The GPT-2 byte-to-unicode table: printable Latin-1 bytes map to
/// themselves, everything else to U+0100 upward.
fn byte_level_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256_u32;
    for byte in 0..=255_u8 {
        let printable = matches!(byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff);
        table[byte as usize] = if printable {
            char::from(byte)
        } else {
            let c = char::from_u32(next).unwrap();
            next += 1;
            c
        };
    }
    table
}

fn metadata_strings<'a>(info: &'a GgufInfo, key: &str) -> Result<Vec<&'a str>, TokenizerError> {
    info.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| err(format!("missing {key}")))?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| err(format!("{key} holds a non-string")))
        })
        .collect()
}

fn metadata_id(info: &GgufInfo, key: &str, vocab: usize) -> Option<TokenId> {
    let id = info.get(key)?.as_u64()?;
    (id < vocab as u64).then_some(id as TokenId)
}

impl Tokenizer {
    pub fn from_gguf(info: &GgufInfo) -> Result<Self, TokenizerError> {
        let model_name = info
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .ok_or_else(|| err("missing tokenizer.ggml.model"))?
            .to_string();
        let tokens: Vec<String> = metadata_strings(info, "tokenizer.ggml.tokens")?
            .into_iter()
            .map(str::to_string)
            .collect();
        let vocab = tokens.len();
        if vocab == 0 || vocab > u32::MAX as usize {
            return Err(err("vocabulary is empty or too large"));
        }
        let types: Vec<TokenType> = match info.get("tokenizer.ggml.token_type") {
            Some(value) => {
                let codes = value
                    .as_array()
                    .ok_or_else(|| err("token_type is not an array"))?;
                if codes.len() != vocab {
                    return Err(err("token_type length differs from vocabulary"));
                }
                codes
                    .iter()
                    .map(|c| c.as_i64().map(TokenType::from_code))
                    .collect::<Option<_>>()
                    .ok_or_else(|| err("token_type holds a non-integer"))?
            }
            None => vec![TokenType::Normal; vocab],
        };
        let mut lookup = HashMap::with_capacity(vocab);
        for (id, text) in tokens.iter().enumerate() {
            // First occurrence wins, matching llama.cpp.
            lookup.entry(text.clone()).or_insert(id as TokenId);
        }

        let mut specials: Vec<(String, TokenId, TokenType)> = tokens
            .iter()
            .zip(&types)
            .enumerate()
            .filter(|(_, (text, kind))| {
                !text.is_empty() && matches!(kind, TokenType::Control | TokenType::UserDefined)
            })
            .map(|(id, (text, kind))| (text.clone(), id as TokenId, *kind))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        let byte_tokens_from = |lookup: &HashMap<String, TokenId>, names: &dyn Fn(u8) -> String| {
            let mut out = [None; 256];
            for byte in 0..=255_u8 {
                out[byte as usize] = lookup.get(&names(byte)).copied();
            }
            out
        };

        let algorithm = match model_name.as_str() {
            "gpt2" => {
                let pre_name = info
                    .get("tokenizer.ggml.pre")
                    .and_then(Value::as_str)
                    .unwrap_or("default");
                let (patterns, ignore_merges) = pre_patterns(pre_name)?;
                let pre = patterns
                    .iter()
                    .map(|p| Regex::new(p).map_err(|e| err(format!("bad regex: {e}"))))
                    .collect::<Result<_, _>>()?;
                let byte_to_char = byte_level_table();
                let char_to_byte = byte_to_char
                    .iter()
                    .enumerate()
                    .map(|(b, &c)| (c, b as u8))
                    .collect();
                let byte_tokens =
                    byte_tokens_from(&lookup, &|b| byte_to_char[b as usize].to_string());
                let merge_lines = metadata_strings(info, "tokenizer.ggml.merges")?;
                let mut merges = HashMap::with_capacity(merge_lines.len());
                for (rank, line) in merge_lines.iter().enumerate() {
                    let Some((left, right)) = line.split_once(' ') else {
                        return Err(err(format!("malformed merge {line:?}")));
                    };
                    let (Some(&l), Some(&r), Some(&m)) = (
                        lookup.get(left),
                        lookup.get(right),
                        lookup.get(&format!("{left}{right}")),
                    ) else {
                        continue; // llama.cpp also skips unusable merges.
                    };
                    merges.entry((l, r)).or_insert((rank as u32, m));
                }
                Algorithm::Bpe {
                    merges,
                    pre,
                    ignore_merges,
                    byte_to_char,
                    char_to_byte,
                    byte_tokens,
                }
            }
            "llama" => {
                let scores = match info.get("tokenizer.ggml.scores").and_then(Value::as_array) {
                    Some(values) if values.len() == vocab => values
                        .iter()
                        .map(|v| v.as_f32().ok_or_else(|| err("score is not a number")))
                        .collect::<Result<_, _>>()?,
                    _ => return Err(err("SentencePiece vocabulary needs tokenizer.ggml.scores")),
                };
                Algorithm::Spm {
                    scores,
                    add_space_prefix: info
                        .get("tokenizer.ggml.add_space_prefix")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                    byte_tokens: byte_tokens_from(&lookup, &|b| format!("<0x{b:02X}>")),
                }
            }
            other => return Err(err(format!("unsupported tokenizer model {other:?}"))),
        };

        let bos = metadata_id(info, "tokenizer.ggml.bos_token_id", vocab);
        let eos = metadata_id(info, "tokenizer.ggml.eos_token_id", vocab);
        let mut eog: Vec<TokenId> = Vec::new();
        for key in [
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id",
        ] {
            if let Some(id) = metadata_id(info, key, vocab) {
                eog.push(id);
            }
        }
        // Common end-of-turn markers that older conversions do not flag.
        for name in [
            "<|im_end|>",
            "<|eot_id|>",
            "<|eom_id|>",
            "<|end|>",
            "<end_of_turn>",
            "<|endoftext|>",
            "</s>",
        ] {
            if let Some(&id) = lookup.get(name)
                && types[id as usize] == TokenType::Control
            {
                eog.push(id);
            }
        }
        eog.sort_unstable();
        eog.dedup();
        let add_bos = info
            .get("tokenizer.ggml.add_bos_token")
            .and_then(Value::as_bool)
            .unwrap_or(model_name == "llama");
        let pre_name = info
            .get("tokenizer.ggml.pre")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        Ok(Self {
            tokens,
            types,
            lookup,
            specials,
            algorithm,
            bos,
            eos,
            eog,
            add_bos,
            model_name,
            pre_name,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_text(&self, id: TokenId) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn token_id(&self, text: &str) -> Option<TokenId> {
        self.lookup.get(text).copied()
    }

    pub fn is_eog(&self, id: TokenId) -> bool {
        self.eog.binary_search(&id).is_ok()
    }

    /// Tokenize `text`. With `parse_special`, control tokens such as
    /// `<|im_start|>` written in the text become their token IDs; otherwise
    /// only user-defined tokens are matched literally.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<TokenId> {
        let mut out = Vec::new();
        if add_special
            && self.add_bos
            && let Some(bos) = self.bos
        {
            out.push(bos);
        }
        let mut first = true;
        for fragment in self.partition(text, parse_special) {
            match fragment {
                Fragment::Token(id) => out.push(id),
                Fragment::Text(piece) => {
                    self.encode_text(piece, first, &mut out);
                }
            }
            first = false;
        }
        out
    }

    fn partition<'t>(&self, text: &'t str, parse_special: bool) -> Vec<Fragment<'t>> {
        let mut fragments = vec![Fragment::Text(text)];
        for (special, id, kind) in &self.specials {
            if !parse_special && *kind == TokenType::Control {
                continue;
            }
            let mut next = Vec::with_capacity(fragments.len());
            for fragment in fragments {
                let Fragment::Text(mut rest) = fragment else {
                    next.push(fragment);
                    continue;
                };
                while let Some(at) = rest.find(special.as_str()) {
                    if at > 0 {
                        next.push(Fragment::Text(&rest[..at]));
                    }
                    next.push(Fragment::Token(*id));
                    rest = &rest[at + special.len()..];
                }
                if !rest.is_empty() {
                    next.push(Fragment::Text(rest));
                }
            }
            fragments = next;
        }
        fragments
    }

    fn encode_text(&self, text: &str, first: bool, out: &mut Vec<TokenId>) {
        match &self.algorithm {
            Algorithm::Bpe {
                merges,
                pre,
                ignore_merges,
                byte_to_char,
                byte_tokens,
                ..
            } => {
                for word in split_sequential(pre, text) {
                    let mapped: String = word.bytes().map(|b| byte_to_char[b as usize]).collect();
                    if *ignore_merges && let Some(&id) = self.lookup.get(&mapped) {
                        out.push(id);
                        continue;
                    }
                    let symbols: Vec<TokenId> = mapped
                        .chars()
                        .map(|c| {
                            let mut buf = [0; 4];
                            self.lookup
                                .get(c.encode_utf8(&mut buf) as &str)
                                .copied()
                                .unwrap_or(u32::MAX)
                        })
                        .collect();
                    if symbols.contains(&u32::MAX) {
                        // A character with no token: fall back to bytes.
                        out.extend(word.bytes().filter_map(|b| byte_tokens[b as usize]));
                        continue;
                    }
                    bpe_merge(symbols, |l, r| merges.get(&(l, r)).copied(), out);
                }
            }
            Algorithm::Spm {
                scores,
                add_space_prefix,
                byte_tokens,
            } => {
                let mut normalized = String::with_capacity(text.len() + 3);
                if *add_space_prefix && first {
                    normalized.push('\u{2581}');
                }
                for c in text.chars() {
                    normalized.push(if c == ' ' { '\u{2581}' } else { c });
                }
                self.spm(&normalized, scores, byte_tokens, out);
            }
        }
    }

    fn spm(
        &self,
        text: &str,
        scores: &[f32],
        byte_tokens: &[Option<TokenId>; 256],
        out: &mut Vec<TokenId>,
    ) {
        // Symbols are byte ranges into `text`; merge the highest-scoring
        // adjacent pair that forms a vocabulary token until none remain.
        let starts: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        if starts.is_empty() {
            return;
        }
        let mut ends: Vec<usize> = starts[1..].to_vec();
        ends.push(text.len());
        let n = starts.len();
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<usize> = (1..=n).collect();
        let mut alive = vec![true; n];
        let mut heap = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<SpmPair>,
                    l: usize,
                    r: usize,
                    starts: &[usize],
                    ends: &[usize]| {
            if let Some(&id) = self.lookup.get(&text[starts[l]..ends[r]]) {
                heap.push(SpmPair {
                    score: scores[id as usize],
                    left: l,
                    right: r,
                    len: ends[r] - starts[l],
                });
            }
        };
        for i in 0..n - 1 {
            push(&mut heap, i, i + 1, &starts, &ends);
        }
        while let Some(pair) = heap.pop() {
            let (l, r) = (pair.left, pair.right);
            if !alive[l] || !alive[r] || next[l] != r || ends[r] - starts[l] != pair.len {
                continue; // Stale entry.
            }
            ends[l] = ends[r];
            alive[r] = false;
            next[l] = next[r];
            if next[l] < n {
                prev[next[l]] = l as isize;
            }
            if prev[l] >= 0 {
                push(&mut heap, prev[l] as usize, l, &starts, &ends);
            }
            if next[l] < n {
                push(&mut heap, l, next[l], &starts, &ends);
            }
        }
        let mut i = 0;
        while i < n {
            let piece = &text[starts[i]..ends[i]];
            match self.lookup.get(piece) {
                Some(&id) => out.push(id),
                None => {
                    for b in piece.bytes() {
                        if let Some(id) = byte_tokens[b as usize] {
                            out.push(id);
                        } else if let Some(unk) = self.unknown() {
                            out.push(unk);
                        }
                    }
                }
            }
            i = next[i];
        }
    }

    fn unknown(&self) -> Option<TokenId> {
        self.types
            .iter()
            .position(|t| *t == TokenType::Unknown)
            .map(|i| i as TokenId)
    }

    /// Bytes a token contributes to the output text. Control tokens decode
    /// to nothing unless `special` is set.
    pub fn token_bytes(&self, id: TokenId, special: bool) -> Vec<u8> {
        let Some(text) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        match self.types[id as usize] {
            TokenType::Control | TokenType::Unknown | TokenType::Unused => {
                if special {
                    text.as_bytes().to_vec()
                } else {
                    Vec::new()
                }
            }
            TokenType::UserDefined => text.as_bytes().to_vec(),
            TokenType::Byte | TokenType::Normal => match &self.algorithm {
                Algorithm::Bpe { char_to_byte, .. } => {
                    let mut out = Vec::with_capacity(text.len());
                    for c in text.chars() {
                        match char_to_byte.get(&c) {
                            Some(&b) => out.push(b),
                            None => {
                                let mut buf = [0; 4];
                                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                            }
                        }
                    }
                    out
                }
                Algorithm::Spm { .. } => {
                    if let Some(hex) = text.strip_prefix("<0x").and_then(|t| t.strip_suffix('>'))
                        && let Ok(b) = u8::from_str_radix(hex, 16)
                    {
                        return vec![b];
                    }
                    text.replace('\u{2581}', " ").into_bytes()
                }
            },
        }
    }

    /// Decode a token sequence to text (lossy for invalid UTF-8).
    pub fn decode(&self, ids: &[TokenId], special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend(self.token_bytes(id, special));
        }
        if let Algorithm::Spm {
            add_space_prefix: true,
            ..
        } = self.algorithm
            && bytes.first() == Some(&b' ')
        {
            bytes.remove(0);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

enum Fragment<'t> {
    Text(&'t str),
    Token(TokenId),
}

/// Apply each regex in turn to every piece produced by the previous one,
/// keeping both matches and the text between them (llama.cpp semantics).
fn split_sequential<'t>(patterns: &[Regex], text: &'t str) -> Vec<&'t str> {
    let mut pieces = vec![text];
    for regex in patterns {
        let mut next = Vec::with_capacity(pieces.len() * 2);
        for piece in pieces {
            let mut last = 0;
            for m in regex.find_iter(piece).flatten() {
                if m.start() > last {
                    next.push(&piece[last..m.start()]);
                }
                if m.end() > m.start() {
                    next.push(m.as_str());
                }
                last = m.end();
            }
            if last < piece.len() {
                next.push(&piece[last..]);
            }
        }
        pieces = next;
    }
    pieces
}

/// Byte-level BPE: repeatedly apply the lowest-ranked merge.
fn bpe_merge(
    mut symbols: Vec<TokenId>,
    merge: impl Fn(TokenId, TokenId) -> Option<(u32, TokenId)>,
    out: &mut Vec<TokenId>,
) {
    loop {
        let mut best: Option<(u32, usize, TokenId)> = None;
        for i in 0..symbols.len().saturating_sub(1) {
            if let Some((rank, merged)) = merge(symbols[i], symbols[i + 1])
                && best.is_none_or(|(r, _, _)| rank < r)
            {
                best = Some((rank, i, merged));
            }
        }
        let Some((_, at, merged)) = best else { break };
        // Merge every occurrence of this pair left to right in one pass.
        let (l, r) = (symbols[at], symbols[at + 1]);
        let mut write = Vec::with_capacity(symbols.len());
        let mut i = 0;
        while i < symbols.len() {
            if i + 1 < symbols.len() && symbols[i] == l && symbols[i + 1] == r {
                write.push(merged);
                i += 2;
            } else {
                write.push(symbols[i]);
                i += 1;
            }
        }
        symbols = write;
    }
    out.extend(symbols);
}

#[derive(Debug)]
struct SpmPair {
    score: f32,
    left: usize,
    right: usize,
    len: usize,
}

impl PartialEq for SpmPair {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for SpmPair {}
impl PartialOrd for SpmPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SpmPair {
    fn cmp(&self, other: &Self) -> Ordering {
        // Highest score first; on ties the leftmost pair wins.
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

/// A tiny ChatML / Llama-3 / Gemma prompt formatter selected from the
/// model's chat template. Jinja is not evaluated; the template is only
/// sniffed for its family.
pub fn format_chat(info: &GgufInfo, system: Option<&str>, user: &str) -> Option<String> {
    let template = info.get("tokenizer.chat_template")?.as_str()?;
    let mut out = String::new();
    if template.contains("<|im_start|>") {
        if let Some(system) = system {
            out += &format!("<|im_start|>system\n{system}<|im_end|>\n");
        }
        out += &format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
    } else if template.contains("<|start_header_id|>") {
        out += "<|begin_of_text|>";
        if let Some(system) = system {
            out += &format!("<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>");
        }
        out += &format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    } else if template.contains("<start_of_turn>") {
        let body = match system {
            Some(system) => format!("{system}\n\n{user}"),
            None => user.to_string(),
        };
        out += &format!("<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n");
    } else if template.contains("[INST]") {
        let body = match system {
            Some(system) => format!("{system}\n\n{user}"),
            None => user.to_string(),
        };
        out += &format!("[INST] {body} [/INST]");
    } else {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_is_a_bijection() {
        let table = byte_level_table();
        let mut seen = std::collections::HashSet::new();
        for c in table {
            assert!(seen.insert(c));
        }
        assert_eq!(table[b'A' as usize], 'A');
        assert_eq!(table[b' ' as usize], '\u{120}'); // "Ġ"
        assert_eq!(table[b'\n' as usize], '\u{10a}'); // "Ċ"
    }

    #[test]
    fn sequential_split_keeps_unmatched_text() {
        let patterns = vec![Regex::new(DIGITS).unwrap(), Regex::new(GPT2).unwrap()];
        assert_eq!(
            split_sequential(&patterns, "ab 12 cd"),
            vec!["ab", " ", "1", "2", " cd"]
        );
    }

    #[test]
    fn gpt2_split_matches_reference_regex_behaviour() {
        let patterns = vec![Regex::new(GPT2).unwrap()];
        assert_eq!(
            split_sequential(&patterns, "Hello  world it's"),
            vec!["Hello", " ", " world", " it", "'s"]
        );
    }
}
