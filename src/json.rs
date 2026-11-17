//! Just enough JSON for the HTTP API.
//!
//! kLLM takes no serialization dependency, so this is a hand-written
//! recursive-descent parser and writer over the subset the Messages API
//! needs. Objects keep insertion order in a `Vec` rather than hashing:
//! request bodies have a handful of keys, and ordered output is easier to
//! read in a packet capture.

use std::fmt::Write as _;

/// The maximum nesting depth a parsed document may reach.
///
/// Parsing is recursive, so a hostile body of 100k open brackets would
/// otherwise overflow the stack of whichever connection thread read it.
const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    /// The value at `key`, if this is an object that has one.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Like [`Value::get`], but `null` reads as absent.
    ///
    /// Clients routinely send `"system": null` or `"stop_sequences": null`
    /// for "not set", and every caller here wants to treat that as a default
    /// rather than as a type error.
    pub fn field(&self, key: &str) -> Option<&Value> {
        self.get(key).filter(|v| !matches!(v, Self::Null))
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// A non-negative integer, rejecting fractions, NaN and overflow.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Self::Number(n) if n.is_finite() && *n >= 0.0 && n.fract() == 0.0 => {
                Some(*n as usize)
            }
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// Parse a complete JSON document, rejecting trailing content.
    pub fn parse(text: &str) -> Result<Self, String> {
        let bytes = text.as_bytes();
        let mut p = Parser { bytes, at: 0 };
        p.skip_whitespace();
        let value = p.value(0)?;
        p.skip_whitespace();
        if p.at != bytes.len() {
            return Err(format!("trailing JSON content at byte {}", p.at));
        }
        Ok(value)
    }

    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(true) => out.push_str("true"),
            Self::Bool(false) => out.push_str("false"),
            Self::Number(n) => write_number(*n, out),
            Self::String(s) => write_string(s, out),
            Self::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Self::Object(fields) => {
                out.push('{');
                for (i, (key, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// `Value::Number` is an `f64`, but the API's integers must not come back as
/// `1024.0` — clients that parse `max_tokens` strictly would reject it.
fn write_number(n: f64, out: &mut String) {
    if !n.is_finite() {
        // JSON has no infinity or NaN. Null is the only honest encoding.
        out.push_str("null");
    } else if n.fract() == 0.0 && n.abs() < 9.007_199_254_740_992e15 {
        let _ = write!(out, "{}", n as i64);
    } else {
        let _ = write!(out, "{n}");
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            // Other controls have no short escape and must not go out raw.
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Build `{"key": value, ...}` without the noise of nested enum literals.
pub fn object<const N: usize>(fields: [(&str, Value); N]) -> Value {
    Value::Object(fields.map(|(k, v)| (k.to_string(), v)).into())
}

pub fn string(s: impl Into<String>) -> Value {
    Value::String(s.into())
}

pub fn number(n: impl Into<f64>) -> Value {
    Value::Number(n.into())
}

/// `usize` has no `Into<f64>` (it is lossy past 2^53), so it needs its own.
pub fn count(n: usize) -> Value {
    Value::Number(n as f64)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Parser<'a> {
    fn skip_whitespace(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.peek() == Some(byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!(
                "expected {:?} at byte {}",
                byte as char, self.at
            ))
        }
    }

    fn literal(&mut self, word: &str) -> Result<(), String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(())
        } else {
            Err(format!("invalid JSON at byte {}", self.at))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err(format!("JSON nested deeper than {MAX_DEPTH}"));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Value::String),
            Some(b't') => self.literal("true").map(|()| Value::Bool(true)),
            Some(b'f') => self.literal("false").map(|()| Value::Bool(false)),
            Some(b'n') => self.literal("null").map(|()| Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(c) => Err(format!("unexpected {:?} at byte {}", c as char, self.at)),
            None => Err("JSON ended early".into()),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, String> {
        self.expect(b'{')?;
        let mut fields = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Value::Object(fields));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.value(depth + 1)?;
            fields.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::Object(fields));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.at)),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.at)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let byte = self.peek().ok_or("string ended early")?;
            self.at += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = self.peek().ok_or("escape ended early")?;
                    self.at += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        c => return Err(format!("bad escape \\{:?}", c as char)),
                    }
                }
                // A raw byte run: copy it as UTF-8 rather than char by char.
                _ => {
                    let start = self.at - 1;
                    while !matches!(self.peek(), None | Some(b'"' | b'\\')) {
                        self.at += 1;
                    }
                    out.push_str(
                        std::str::from_utf8(&self.bytes[start..self.at])
                            .map_err(|_| "string is not valid UTF-8".to_string())?,
                    );
                }
            }
        }
    }

    /// `\uXXXX`, joining a surrogate pair into the character it encodes.
    fn unicode_escape(&mut self) -> Result<char, String> {
        let high = self.hex4()?;
        // Lone low surrogates, and highs with no low, cannot be represented;
        // substitute U+FFFD rather than failing the whole request.
        if !(0xd800..0xdc00).contains(&high) {
            return Ok(char::from_u32(high).unwrap_or('\u{fffd}'));
        }
        if !self.bytes[self.at..].starts_with(b"\\u") {
            return Ok('\u{fffd}');
        }
        let save = self.at;
        self.at += 2;
        let low = self.hex4()?;
        if !(0xdc00..0xe000).contains(&low) {
            self.at = save;
            return Ok('\u{fffd}');
        }
        let combined = 0x1_0000 + ((high - 0xd800) << 10) + (low - 0xdc00);
        Ok(char::from_u32(combined).unwrap_or('\u{fffd}'))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .ok_or("truncated \\u escape")?;
        let text = std::str::from_utf8(digits).map_err(|_| "bad \\u escape".to_string())?;
        let value = u32::from_str_radix(text, 16).map_err(|_| "bad \\u escape".to_string())?;
        self.at += 4;
        Ok(value)
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        while matches!(
            self.peek(),
            Some(b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
        ) {
            self.at += 1;
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .ok()
            .and_then(|t| t.parse::<f64>().ok())
            .map(Value::Number)
            .ok_or_else(|| format!("bad number at byte {start}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_nested_document() {
        let text = r#"{"a":[1,2.5,true,null],"b":{"c":"hi"}}"#;
        let value = Value::parse(text).unwrap();
        assert_eq!(value.to_json(), text);
    }

    #[test]
    fn integers_do_not_gain_a_decimal_point() {
        assert_eq!(number(1024.0).to_json(), "1024");
        assert_eq!(count(0).to_json(), "0");
        assert_eq!(number(0.5).to_json(), "0.5");
    }

    #[test]
    fn reads_escapes_including_surrogate_pairs() {
        let value = Value::parse(r#""a\nA😀\"""#).unwrap();
        assert_eq!(value.as_str().unwrap(), "a\nA\u{1f600}\"");
    }

    #[test]
    fn lone_surrogate_becomes_a_replacement_char() {
        let value = Value::parse(r#""\ud83d""#).unwrap();
        assert_eq!(value.as_str().unwrap(), "\u{fffd}");
    }

    #[test]
    fn escapes_control_characters_on_the_way_out() {
        assert_eq!(string("a\u{1}b").to_json(), r#""a\u0001b""#);
    }

    #[test]
    fn null_fields_read_as_absent() {
        let value = Value::parse(r#"{"system":null,"max_tokens":8}"#).unwrap();
        assert!(value.field("system").is_none());
        assert_eq!(value.field("max_tokens").unwrap().as_usize(), Some(8));
    }

    #[test]
    fn rejects_trailing_content_and_deep_nesting() {
        assert!(Value::parse("{} {}").is_err());
        let deep = "[".repeat(MAX_DEPTH + 2);
        assert!(Value::parse(&deep).is_err());
    }

    #[test]
    fn fractional_values_are_not_counts() {
        assert_eq!(Value::Number(1.5).as_usize(), None);
        assert_eq!(Value::Number(-1.0).as_usize(), None);
    }
}
