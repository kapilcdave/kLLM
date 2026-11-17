//! An Anthropic Messages API server, so Claude Code can drive a local model.
//!
//! `POST /v1/messages`, streaming and not, plus `/v1/messages/count_tokens`
//! and `/v1/models`. The wire format is Anthropic's because that is what
//! Claude Code speaks; the prompt underneath is ChatML, which is what the
//! Qwen models kLLM runs were trained on.
//!
//! Three things shape the design:
//!
//! * **One model, one session.** Decoding is already using every core, so
//!   concurrency would only make both requests slower. HTTP is handled on a
//!   thread per connection — an idle keep-alive connection must not block a
//!   new one — but generation serializes on a mutex.
//! * **Prefix reuse is not an optimisation here, it is the feature.** A chat
//!   client resends the entire conversation on every turn. At this machine's
//!   prefill speed, re-reading a 10k-token history would cost half a minute
//!   per turn, so the session is kept alive and only the new tail is fed.
//! * **Tools go through the model's own format.** Qwen3 was trained to emit
//!   `<tool_call>{...}</tool_call>`, so Anthropic's `tools` are rendered into
//!   the system prompt in that shape and the output is translated back into
//!   `tool_use` blocks.

use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Instant,
};

use crate::json::{Value, count, object, string};
use crate::model::{Logits, Model, Session};
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::{Tokenizer, Utf8Stream};

/// Cap on a request body. Long contexts make these genuinely large — a 200k
/// token conversation is megabytes of JSON — but not unbounded.
const MAX_BODY: usize = 128 * 1024 * 1024;

/// How many connections may be open at once. Generation is serialized
/// anyway; this only bounds the threads a client can make us spawn.
const MAX_CONNECTIONS: usize = 64;

/// The markers Qwen3 was trained to wrap a tool call in.
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

pub struct Options {
    pub host: String,
    pub port: u16,
    /// When set, requests must present it as `x-api-key` or a bearer token.
    pub api_key: Option<String>,
    /// Used when the client omits `max_tokens`.
    pub default_max_tokens: usize,
    /// Sampling used when the client does not ask for something else.
    pub sampler: SamplerConfig,
    /// Log one line per request.
    pub verbose: bool,
}

/// What went wrong, and therefore what the client should be told.
enum Fail {
    /// The request cannot be served as written: 400.
    Request(String),
    /// Generation failed: 500.
    Engine(String),
    /// The client hung up mid-response. Nothing left to say.
    Gone,
}

impl From<io::Error> for Fail {
    fn from(_: io::Error) -> Self {
        Self::Gone
    }
}

/// The model, its tokenizer, and the live session they share.
pub struct Engine {
    tokenizer: Tokenizer,
    session: Session<'static>,
    /// Tokens the session has already processed. Kept exactly in step with
    /// `session.position()` so a prefix match can be trusted.
    cached: Vec<u32>,
    context: usize,
    batch: usize,
    /// Whether to let a reasoning model reason, or pre-fill an empty
    /// `<think></think>` the way its own template does.
    think: bool,
    /// Whether the template knows `<tool_call>`. Without it, injecting tool
    /// definitions would be teaching the model a format it never saw.
    tool_format: bool,
    name: String,
    /// Prompt tokens the last request reused from the session, and how many
    /// it still had to read. Reported per request because a client that
    /// perturbs its own prefix turns every turn into a full prefill, and the
    /// only way to notice is to watch these two numbers.
    reused: usize,
    fed: usize,
}

impl Engine {
    /// `model` is borrowed for the process's life: the session holds it, and
    /// the server never unloads.
    pub fn new(
        model: &'static Model,
        context: usize,
        batch: usize,
        allow_think: bool,
    ) -> Result<Self, String> {
        let info = model.info();
        let template = info
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if !template.contains("<|im_start|>") {
            return Err(
                "kllm serve renders ChatML (<|im_start|>); this model uses another template".into(),
            );
        }
        let tokenizer = Tokenizer::from_gguf(info).map_err(|e| e.to_string())?;
        let session = model
            .new_session(context, batch)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            tokenizer,
            session,
            cached: Vec::new(),
            context,
            batch,
            think: allow_think && template.contains("<think>"),
            tool_format: template.contains(TOOL_OPEN),
            name: info
                .get("general.name")
                .and_then(|v| v.as_str())
                .unwrap_or("kllm")
                .to_string(),
            reused: 0,
            fed: 0,
        })
    }

    pub fn describe(&self) -> &str {
        &self.name
    }

    /// Drop the whole cache. Used when an error leaves the session's position
    /// and `cached` possibly disagreeing — a stale prefix match would feed
    /// the model a history it never saw.
    fn reset(&mut self) {
        self.session.truncate(0);
        self.cached.clear();
    }

    fn count_tokens(&self, prompt: &str) -> usize {
        self.tokenizer.encode(prompt, true, true).len()
    }
}

/// One `/v1/messages` call reduced to what generation needs.
struct Request {
    prompt: String,
    /// Echoed back, because clients match the response against what they
    /// asked for. kLLM runs whatever model it was started with.
    model: String,
    max_tokens: usize,
    stream: bool,
    stop: Vec<String>,
    sampler: SamplerConfig,
    /// The client offered tools, so a `<tool_call>` is expected output
    /// rather than text that happens to look like one.
    tools: bool,
}

/// An event on the way out, consumed either by the SSE writer or by the
/// buffer that builds a single JSON response.
enum Delta<'a> {
    /// The prompt is in and `input_tokens` is final.
    Prompt(usize),
    Text(&'a str),
    Tool {
        id: String,
        name: String,
        /// The arguments object, already serialized.
        input: String,
    },
    Done {
        stop_reason: &'static str,
        stop_sequence: Option<String>,
        output_tokens: usize,
    },
}

// ---------------------------------------------------------------------------
// Prompt rendering
// ---------------------------------------------------------------------------

/// The text of a `content` field, which the API allows as a bare string or
/// as an array of typed blocks.
fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut out = String::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = block.get("text").and_then(Value::as_str) {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(t);
                        }
                    }
                    // A local text model cannot see these. Saying so beats
                    // dropping them silently and leaving the model to answer
                    // a question about an image it was never shown.
                    Some("image") => {
                        out.push_str("\n[an image was attached; this model cannot read images]")
                    }
                    Some("document") => out
                        .push_str("\n[a document was attached; this model cannot read documents]"),
                    _ => {}
                }
            }
            out
        }
        _ => String::new(),
    }
}

/// Renders `tools` as Qwen3's own template does, so the model sees the
/// format it was trained on rather than a paraphrase.
fn render_tools(tools: &[Value], out: &mut String) {
    out.push_str(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    for tool in tools {
        let name = tool.get("name").and_then(Value::as_str).unwrap_or_default();
        let description = tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default();
        // Anthropic calls the schema `input_schema`; Qwen expects an OpenAI
        // style `function` wrapper with `parameters`.
        let parameters = tool
            .field("input_schema")
            .cloned()
            .unwrap_or_else(|| object([("type", string("object"))]));
        let signature = object([
            ("type", string("function")),
            (
                "function",
                object([
                    ("name", string(name)),
                    ("description", string(description)),
                    ("parameters", parameters),
                ]),
            ),
        ]);
        out.push('\n');
        out.push_str(&signature.to_json());
    }
    out.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name and \
         arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
}

impl Engine {
    /// Turns an Anthropic request body into a ChatML prompt.
    ///
    /// The rendering is deliberately a pure function of the conversation so
    /// that adding a turn extends the previous prompt rather than perturbing
    /// it — which is what makes the prefix cache hit.
    fn render(&self, body: &Value) -> Result<(String, bool), String> {
        let messages = body
            .field("messages")
            .and_then(Value::as_array)
            .ok_or("`messages` is required and must be an array")?;
        let tools: &[Value] = body
            .field("tools")
            .and_then(Value::as_array)
            .unwrap_or_default();
        let use_tools = !tools.is_empty() && self.tool_format;

        let system = body.field("system").map(text_of).unwrap_or_default();
        let mut out = String::new();
        if !system.is_empty() || use_tools {
            out.push_str("<|im_start|>system\n");
            out.push_str(&system);
            if use_tools {
                render_tools(tools, &mut out);
            }
            out.push_str("<|im_end|>\n");
        }

        for message in messages {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .ok_or("every message needs a `role`")?;
            let content = message.get("content").unwrap_or(&Value::Null);
            match role {
                "user" => self.render_user(content, &mut out),
                "assistant" => self.render_assistant(content, &mut out),
                // Not in Anthropic's documented schema -- system belongs in
                // the top-level field -- but Claude Code sends one, and ChatML
                // has a system turn, so render it where it was placed rather
                // than rejecting the request.
                "system" => {
                    let text = text_of(content);
                    if !text.is_empty() {
                        out.push_str("<|im_start|>system\n");
                        out.push_str(&text);
                        out.push_str("<|im_end|>\n");
                    }
                }
                other => return Err(format!("unsupported message role {other:?}")),
            }
        }

        out.push_str("<|im_start|>assistant\n");
        if !self.think {
            // What Qwen3's own template emits for `enable_thinking=false`.
            out.push_str("<think>\n\n</think>\n\n");
        }
        Ok((out, use_tools))
    }

    /// Anthropic nests tool results inside a user message; Qwen3 expects each
    /// as its own turn wrapped in `<tool_response>`, so they are split out.
    fn render_user(&self, content: &Value, out: &mut String) {
        if let Value::Array(blocks) = content {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                    continue;
                }
                let body = block.get("content").map(text_of).unwrap_or_default();
                let body = match block.get("is_error").and_then(Value::as_bool) {
                    Some(true) => format!("Error: {body}"),
                    _ => body,
                };
                out.push_str("<|im_start|>user\n<tool_response>\n");
                out.push_str(&body);
                out.push_str("\n</tool_response><|im_end|>\n");
            }
        }
        let text = text_of(content);
        if !text.is_empty() {
            out.push_str("<|im_start|>user\n");
            out.push_str(&text);
            out.push_str("<|im_end|>\n");
        }
    }

    fn render_assistant(&self, content: &Value, out: &mut String) {
        out.push_str("<|im_start|>assistant\n");
        out.push_str(&text_of(content));
        if let Value::Array(blocks) = content {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                    continue;
                }
                let call = object([
                    (
                        "name",
                        string(
                            block
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                        ),
                    ),
                    (
                        "arguments",
                        block
                            .field("input")
                            .cloned()
                            .unwrap_or_else(|| Value::Object(Vec::new())),
                    ),
                ]);
                out.push('\n');
                out.push_str(TOOL_OPEN);
                out.push('\n');
                out.push_str(&call.to_json());
                out.push('\n');
                out.push_str(TOOL_CLOSE);
            }
        }
        out.push_str("<|im_end|>\n");
    }

    fn parse_request(&self, body: &Value, opts: &Options) -> Result<Request, String> {
        let (prompt, tools) = self.render(body)?;
        let mut sampler = opts.sampler.clone();
        if let Some(t) = body.field("temperature").and_then(Value::as_f64) {
            sampler.temperature = t as f32;
        }
        if let Some(p) = body.field("top_p").and_then(Value::as_f64) {
            sampler.top_p = p as f32;
        }
        if let Some(k) = body.field("top_k").and_then(Value::as_usize) {
            sampler.top_k = k;
        }
        let stop = body
            .field("stop_sequences")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        Ok(Request {
            prompt,
            model: body
                .field("model")
                .and_then(Value::as_str)
                .unwrap_or(&self.name)
                .to_string(),
            max_tokens: body
                .field("max_tokens")
                .and_then(Value::as_usize)
                .filter(|n| *n > 0)
                .unwrap_or(opts.default_max_tokens),
            stream: body
                .field("stream")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            stop,
            sampler,
            tools,
        })
    }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// How many trailing bytes of `text` must be withheld because they could
/// still grow into one of `markers`.
///
/// Without this, a stop sequence of `"</done>"` would be streamed to the
/// client one character at a time before we ever recognised it.
fn held_back(text: &str, markers: &[String]) -> usize {
    let mut held = 0;
    for marker in markers {
        // Proper prefixes only: a complete marker is a match, not a hold.
        // `char_indices` keeps the split on a character boundary even if the
        // client sent a multi-byte stop sequence.
        for (end, _) in marker.char_indices().skip(1) {
            if text.ends_with(&marker[..end]) {
                held = held.max(end);
            }
        }
    }
    held.min(text.len())
}

/// The earliest complete marker at or after `from`, and which one it was.
fn first_marker(text: &str, from: usize, markers: &[String]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (which, marker) in markers.iter().enumerate() {
        if let Some(offset) = text[from..].find(marker.as_str()) {
            let at = from + offset;
            if best.is_none_or(|(found, _)| at < found) {
                best = Some((at, which));
            }
        }
    }
    best
}

/// Tool-call ids have to be unique within a conversation, and the client
/// quotes them back to us in `tool_result`.
fn next_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}{:016x}", n ^ 0x9e37_79b9_7f4a_7c15)
}

/// Reads the JSON Qwen3 puts between the `<tool_call>` markers.
///
/// Returns `None` when it is not a tool call after all — the model
/// occasionally writes the marker around prose. The caller then shows the
/// text instead of inventing a nameless `tool_use` block the client would
/// have to reject.
fn parse_tool_call(body: &str) -> Option<(String, String)> {
    let value = Value::parse(body.trim()).ok()?;
    let name = value.get("name").and_then(Value::as_str)?;
    if name.is_empty() {
        return None;
    }
    let input = value
        .field("arguments")
        .filter(|a| matches!(a, Value::Object(_)))
        .map(Value::to_json)
        .unwrap_or_else(|| "{}".to_string());
    Some((name.to_string(), input))
}

impl Engine {
    /// Feeds the prompt and decodes a reply, reporting progress through
    /// `emit` so the caller can stream it or buffer it.
    fn generate(
        &mut self,
        request: &Request,
        emit: &mut dyn FnMut(Delta<'_>) -> io::Result<()>,
    ) -> Result<(), Fail> {
        let tokens = self.tokenizer.encode(&request.prompt, true, true);
        if tokens.len() + 1 >= self.context {
            return Err(Fail::Request(format!(
                "the conversation is {} tokens, which does not leave room to reply inside this \
                 server's context of {}; restart kllm serve with a larger --context",
                tokens.len(),
                self.context
            )));
        }
        let budget = request.max_tokens.min(self.context - tokens.len());

        // Reuse whatever of the session still matches. One token must always
        // be forwarded, because sampling needs logits to sample from.
        let shared = common_prefix(&self.cached, &tokens).min(tokens.len() - 1);
        self.session.truncate(shared);
        self.cached.truncate(shared);
        self.reused = shared;
        self.fed = tokens.len() - shared;

        let mut logits = Vec::new();
        for chunk in tokens[shared..].chunks(self.batch) {
            logits = match self.session.forward(chunk, Logits::Last) {
                Ok(values) => values.to_vec(),
                Err(e) => {
                    self.reset();
                    return Err(Fail::Engine(e.to_string()));
                }
            };
        }
        self.cached.extend_from_slice(&tokens[shared..]);
        emit(Delta::Prompt(tokens.len()))?;

        // User stop sequences first, then the tool marker, so a marker index
        // identifies which of the two kinds matched.
        let mut markers = request.stop.clone();
        let tool_marker = (request.tools && self.tool_format).then(|| {
            markers.push(TOOL_OPEN.to_string());
            markers.len() - 1
        });

        let mut sampler = Sampler::new(request.sampler.clone());
        let mut history = tokens;
        let mut utf8 = Utf8Stream::default();
        let mut text = String::new();
        // How much of `text` has been handed to `emit` or deliberately dropped.
        let mut shown = 0usize;
        // Where the body of an open tool call starts, if one is open.
        let mut tool_at: Option<usize> = None;
        let mut tool_calls = 0usize;
        let mut generated = 0usize;
        let mut stop_reason = "end_turn";
        let mut stop_sequence = None;

        'decode: while generated < budget && self.session.position() < self.context {
            let next = sampler.sample(&logits, &history);
            if self.tokenizer.is_eog(next) {
                break;
            }
            history.push(next);
            self.cached.push(next);
            generated += 1;
            text.push_str(&utf8.push(&self.tokenizer.token_bytes(next, false)));

            // One token can finish a marker and begin the next thing, so keep
            // draining until there is nothing more to recognise.
            loop {
                match tool_at {
                    None => {
                        if let Some((at, which)) = first_marker(&text, shown, &markers) {
                            if at > shown {
                                emit(Delta::Text(&text[shown..at]))?;
                            }
                            shown = at;
                            if Some(which) == tool_marker {
                                shown = at + TOOL_OPEN.len();
                                tool_at = Some(shown);
                                continue;
                            }
                            // The stop sequence itself is not part of the
                            // reply, and neither is anything after it.
                            stop_reason = "stop_sequence";
                            stop_sequence = Some(markers[which].clone());
                            break 'decode;
                        }
                        let safe = text.len() - held_back(&text[shown..], &markers);
                        if safe > shown {
                            emit(Delta::Text(&text[shown..safe]))?;
                            shown = safe;
                        }
                        break;
                    }
                    Some(start) => {
                        let Some(offset) = text[start..].find(TOOL_CLOSE) else {
                            break;
                        };
                        let at = start + offset;
                        match parse_tool_call(&text[start..at]) {
                            Some((name, input)) => {
                                emit(Delta::Tool {
                                    id: next_id("toolu_"),
                                    name,
                                    input,
                                })?;
                                tool_calls += 1;
                            }
                            // Not really a call: show what the model wrote.
                            None => {
                                emit(Delta::Text(&text[start - TOOL_OPEN.len()..at]))?;
                            }
                        }
                        shown = at + TOOL_CLOSE.len();
                        tool_at = None;
                        continue;
                    }
                }
            }

            logits = match self.session.forward(&[next], Logits::Last) {
                Ok(values) => values.to_vec(),
                Err(e) => {
                    self.reset();
                    return Err(Fail::Engine(e.to_string()));
                }
            };
        }

        if stop_sequence.is_none() {
            // Nothing more is coming, so bytes held against a marker that
            // never completed are just text. Flush the decoder too.
            let tail = utf8.push(&[]);
            text.push_str(&tail);
            let rest = match tool_at {
                // A tool call cut off by the budget. Emitting the raw text is
                // more useful than a silent truncation.
                Some(start) => start - TOOL_OPEN.len(),
                None => shown,
            };
            if rest < text.len() {
                emit(Delta::Text(&text[rest..]))?;
            }
            if generated >= budget {
                stop_reason = "max_tokens";
            } else if tool_calls > 0 {
                stop_reason = "tool_use";
            }
        }

        emit(Delta::Done {
            stop_reason,
            stop_sequence,
            output_tokens: generated,
        })?;
        Ok(())
    }
}

/// How many leading tokens `a` and `b` share.
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

// ---------------------------------------------------------------------------
// Response shapes
// ---------------------------------------------------------------------------

fn message_envelope(id: &str, model: &str, input_tokens: usize) -> Value {
    object([
        ("id", string(id)),
        ("type", string("message")),
        ("role", string("assistant")),
        ("model", string(model)),
        ("content", Value::Array(Vec::new())),
        ("stop_reason", Value::Null),
        ("stop_sequence", Value::Null),
        (
            "usage",
            object([
                ("input_tokens", count(input_tokens)),
                ("output_tokens", count(0)),
            ]),
        ),
    ])
}

fn error_body(kind: &str, message: &str) -> Value {
    object([
        ("type", string("error")),
        (
            "error",
            object([("type", string(kind)), ("message", string(message))]),
        ),
    ])
}

/// Collects deltas into a single non-streaming response.
#[derive(Default)]
struct Buffered {
    blocks: Vec<Value>,
    /// Index of the open text block, so consecutive text deltas append.
    text: Option<usize>,
    input_tokens: usize,
    output_tokens: usize,
    stop_reason: String,
    stop_sequence: Option<String>,
}

impl Buffered {
    fn push(&mut self, delta: Delta<'_>) {
        match delta {
            Delta::Prompt(n) => self.input_tokens = n,
            Delta::Text(chunk) => match self.text {
                Some(at) => {
                    if let Value::Object(fields) = &mut self.blocks[at]
                        && let Some((_, Value::String(s))) =
                            fields.iter_mut().find(|(k, _)| k == "text")
                    {
                        s.push_str(chunk);
                    }
                }
                None => {
                    self.text = Some(self.blocks.len());
                    self.blocks
                        .push(object([("type", string("text")), ("text", string(chunk))]));
                }
            },
            Delta::Tool { id, name, input } => {
                self.text = None;
                self.blocks.push(object([
                    ("type", string("tool_use")),
                    ("id", string(id)),
                    ("name", string(name)),
                    (
                        "input",
                        Value::parse(&input).unwrap_or(Value::Object(Vec::new())),
                    ),
                ]));
            }
            Delta::Done {
                stop_reason,
                stop_sequence,
                output_tokens,
            } => {
                self.stop_reason = stop_reason.to_string();
                self.stop_sequence = stop_sequence;
                self.output_tokens = output_tokens;
            }
        }
    }

    fn finish(self, id: &str, model: &str) -> Value {
        object([
            ("id", string(id)),
            ("type", string("message")),
            ("role", string("assistant")),
            ("model", string(model)),
            ("content", Value::Array(self.blocks)),
            ("stop_reason", string(self.stop_reason)),
            (
                "stop_sequence",
                self.stop_sequence.map_or(Value::Null, string),
            ),
            (
                "usage",
                object([
                    ("input_tokens", count(self.input_tokens)),
                    ("output_tokens", count(self.output_tokens)),
                ]),
            ),
        ])
    }
}

/// Writes deltas as the Messages API's server-sent event sequence.
struct Streamed<'a> {
    out: &'a mut TcpStream,
    id: String,
    model: String,
    /// Index of the current content block, as the API numbers them.
    index: usize,
    /// Whether a block is open, and whether it is the text one.
    open: Option<bool>,
}

impl Streamed<'_> {
    fn event(&mut self, name: &str, data: &Value) -> io::Result<()> {
        // SSE frames inside HTTP chunked encoding: the chunking keeps the
        // connection reusable, which a bare `Connection: close` stream does not.
        write_chunk(
            self.out,
            format!("event: {name}\ndata: {}\n\n", data.to_json()).as_bytes(),
        )
    }

    fn close_block(&mut self) -> io::Result<()> {
        if self.open.take().is_some() {
            let index = self.index;
            self.event(
                "content_block_stop",
                &object([
                    ("type", string("content_block_stop")),
                    ("index", count(index)),
                ]),
            )?;
            self.index += 1;
        }
        Ok(())
    }

    fn push(&mut self, delta: Delta<'_>) -> io::Result<()> {
        match delta {
            Delta::Prompt(n) => {
                let message = message_envelope(&self.id.clone(), &self.model.clone(), n);
                self.event(
                    "message_start",
                    &object([("type", string("message_start")), ("message", message)]),
                )
            }
            Delta::Text(chunk) => {
                if self.open != Some(true) {
                    self.close_block()?;
                    let index = self.index;
                    self.event(
                        "content_block_start",
                        &object([
                            ("type", string("content_block_start")),
                            ("index", count(index)),
                            (
                                "content_block",
                                object([("type", string("text")), ("text", string(""))]),
                            ),
                        ]),
                    )?;
                    self.open = Some(true);
                }
                let index = self.index;
                self.event(
                    "content_block_delta",
                    &object([
                        ("type", string("content_block_delta")),
                        ("index", count(index)),
                        (
                            "delta",
                            object([("type", string("text_delta")), ("text", string(chunk))]),
                        ),
                    ]),
                )
            }
            Delta::Tool { id, name, input } => {
                self.close_block()?;
                let index = self.index;
                self.event(
                    "content_block_start",
                    &object([
                        ("type", string("content_block_start")),
                        ("index", count(index)),
                        (
                            "content_block",
                            object([
                                ("type", string("tool_use")),
                                ("id", string(id)),
                                ("name", string(name)),
                                ("input", Value::Object(Vec::new())),
                            ]),
                        ),
                    ]),
                )?;
                self.open = Some(false);
                // The arguments are only valid JSON once complete, so they go
                // out as one `input_json_delta` rather than token by token.
                self.event(
                    "content_block_delta",
                    &object([
                        ("type", string("content_block_delta")),
                        ("index", count(index)),
                        (
                            "delta",
                            object([
                                ("type", string("input_json_delta")),
                                ("partial_json", string(input)),
                            ]),
                        ),
                    ]),
                )?;
                self.close_block()
            }
            Delta::Done {
                stop_reason,
                stop_sequence,
                output_tokens,
            } => {
                self.close_block()?;
                self.event(
                    "message_delta",
                    &object([
                        ("type", string("message_delta")),
                        (
                            "delta",
                            object([
                                ("stop_reason", string(stop_reason)),
                                ("stop_sequence", stop_sequence.map_or(Value::Null, string)),
                            ]),
                        ),
                        ("usage", object([("output_tokens", count(output_tokens))])),
                    ]),
                )?;
                self.event("message_stop", &object([("type", string("message_stop"))]))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

struct HttpRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

fn write_chunk(out: &mut TcpStream, data: &[u8]) -> io::Result<()> {
    // A zero-length chunk terminates the body, so never send an empty one.
    if data.is_empty() {
        return Ok(());
    }
    write!(out, "{:x}\r\n", data.len())?;
    out.write_all(data)?;
    out.write_all(b"\r\n")?;
    out.flush()
}

fn write_response(out: &mut TcpStream, status: &str, body: &[u8]) -> io::Result<()> {
    write!(
        out,
        "HTTP/1.1 {status}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: keep-alive\r\n\r\n",
        body.len()
    )?;
    out.write_all(body)?;
    out.flush()
}

fn write_error(out: &mut TcpStream, status: &str, kind: &str, message: &str) -> io::Result<()> {
    write_response(out, status, error_body(kind, message).to_json().as_bytes())
}

/// Reads one request, or `None` when the client closed the connection.
fn read_request(
    reader: &mut BufReader<TcpStream>,
    out: &mut TcpStream,
) -> io::Result<Option<HttpRequest>> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        // Tolerate stray blank lines between pipelined requests.
        if !line.trim().is_empty() {
            break;
        }
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or("/").to_string();

    let mut headers = Vec::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        let header = header.trim_end_matches(['\r', '\n']);
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    let find = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    };

    // Some clients wait for this before sending a large body.
    if find("expect").is_some_and(|v| v.eq_ignore_ascii_case("100-continue")) {
        out.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        out.flush()?;
    }

    let chunked = find("transfer-encoding").is_some_and(|v| v.contains("chunked"));
    let body = if chunked {
        read_chunked_body(reader)?
    } else {
        let length = find("content-length")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        if length > MAX_BODY {
            return Err(io::Error::other("request body too large"));
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body)?;
        body
    };

    Ok(Some(HttpRequest {
        method,
        path: target.split('?').next().unwrap_or("/").to_string(),
        headers,
        body,
    }))
}

fn read_chunked_body(reader: &mut BufReader<TcpStream>) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(io::Error::other("chunked body ended early"));
        }
        // A chunk size may carry extensions after a semicolon.
        let size = line.trim().split(';').next().unwrap_or("0");
        let size =
            usize::from_str_radix(size, 16).map_err(|_| io::Error::other("bad chunk size"))?;
        if size == 0 {
            // Consume the trailer section.
            loop {
                let mut trailer = String::new();
                if reader.read_line(&mut trailer)? == 0 || trailer.trim().is_empty() {
                    break;
                }
            }
            return Ok(body);
        }
        if body.len() + size > MAX_BODY {
            return Err(io::Error::other("request body too large"));
        }
        let start = body.len();
        body.resize(start + size, 0);
        reader.read_exact(&mut body[start..])?;
        let mut crlf = [0; 2];
        reader.read_exact(&mut crlf)?;
    }
}

/// Starts the server and serves until the listener fails.
pub fn serve(engine: Engine, options: Options) -> Result<(), String> {
    let listener = TcpListener::bind((options.host.as_str(), options.port))
        .map_err(|e| format!("cannot bind {}:{}: {e}", options.host, options.port))?;
    let address = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| format!("{}:{}", options.host, options.port));
    eprintln!("kllm: serving {} on http://{address}", engine.describe());
    eprintln!("kllm: Anthropic Messages API at POST /v1/messages (streaming and not)");
    eprintln!("kllm: point Claude Code at it with:");
    eprintln!("kllm:   ANTHROPIC_BASE_URL=http://{address} ANTHROPIC_AUTH_TOKEN=local claude");

    let engine = Arc::new(Mutex::new(engine));
    let options = Arc::new(options);
    let live = Arc::new(AtomicUsize::new(0));

    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        if live.load(Ordering::Relaxed) >= MAX_CONNECTIONS {
            // Refuse rather than spawn without bound. Generation is
            // serialized anyway, so a backlog this deep is a client bug.
            let mut stream = stream;
            let _ = write_error(
                &mut stream,
                "529 Too Many Requests",
                "overloaded_error",
                "too many open connections",
            );
            continue;
        }
        let (engine, options, live) = (engine.clone(), options.clone(), live.clone());
        live.fetch_add(1, Ordering::Relaxed);
        std::thread::spawn(move || {
            handle_connection(stream, &engine, &options);
            live.fetch_sub(1, Ordering::Relaxed);
        });
    }
    Ok(())
}

fn handle_connection(stream: TcpStream, engine: &Mutex<Engine>, options: &Options) {
    // SSE deltas are small and latency is the whole point of streaming.
    let _ = stream.set_nodelay(true);
    let Ok(clone) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(clone);
    let mut out = stream;
    loop {
        match read_request(&mut reader, &mut out) {
            Ok(Some(request)) => {
                if route(&request, &mut out, engine, options).is_err() {
                    return;
                }
            }
            // Clean close, or a client that went away mid-request.
            Ok(None) | Err(_) => return,
        }
    }
}

fn route(
    request: &HttpRequest,
    out: &mut TcpStream,
    engine: &Mutex<Engine>,
    options: &Options,
) -> io::Result<()> {
    // Liveness is exempt from auth. A supervisor or a start-up wait loop has
    // no business holding the key, and the answer reveals nothing.
    if request.method == "GET" && request.path == "/health" {
        return write_response(out, "200 OK", br#"{"status":"ok"}"#);
    }
    if let Some(expected) = &options.api_key {
        let presented = request.header("x-api-key").map(str::to_string).or_else(|| {
            request
                .header("authorization")
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(str::to_string)
        });
        if presented.as_deref() != Some(expected.as_str()) {
            return write_error(
                out,
                "401 Unauthorized",
                "authentication_error",
                "invalid x-api-key",
            );
        }
    }

    match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v1/messages") => messages(request, out, engine, options),
        ("POST", "/v1/messages/count_tokens") => count_tokens(request, out, engine, options),
        ("GET", "/v1/models") => {
            let engine = engine.lock().unwrap_or_else(|e| e.into_inner());
            let model = object([
                ("type", string("model")),
                ("id", string(engine.describe())),
                ("display_name", string(engine.describe())),
            ]);
            drop(engine);
            write_response(
                out,
                "200 OK",
                object([
                    ("data", Value::Array(vec![model])),
                    ("has_more", Value::Bool(false)),
                ])
                .to_json()
                .as_bytes(),
            )
        }
        // Preflight, in case the user points a browser tool at this.
        ("OPTIONS", _) => write_response(out, "204 No Content", b""),
        _ => write_error(
            out,
            "404 Not Found",
            "not_found_error",
            &format!("{} {} is not served", request.method, request.path),
        ),
    }
}

/// Parses the body, or writes the 400 that says why not.
fn body_of(request: &HttpRequest, out: &mut TcpStream) -> io::Result<Option<Value>> {
    let text = match std::str::from_utf8(&request.body) {
        Ok(text) => text,
        Err(_) => {
            write_error(
                out,
                "400 Bad Request",
                "invalid_request_error",
                "request body is not UTF-8",
            )?;
            return Ok(None);
        }
    };
    match Value::parse(text) {
        Ok(value) => Ok(Some(value)),
        Err(e) => {
            write_error(out, "400 Bad Request", "invalid_request_error", &e)?;
            Ok(None)
        }
    }
}

fn count_tokens(
    request: &HttpRequest,
    out: &mut TcpStream,
    engine: &Mutex<Engine>,
    options: &Options,
) -> io::Result<()> {
    let Some(body) = body_of(request, out)? else {
        return Ok(());
    };
    let engine = engine.lock().unwrap_or_else(|e| e.into_inner());
    match engine.parse_request(&body, options) {
        Ok(parsed) => {
            let tokens = engine.count_tokens(&parsed.prompt);
            drop(engine);
            write_response(
                out,
                "200 OK",
                object([("input_tokens", count(tokens))])
                    .to_json()
                    .as_bytes(),
            )
        }
        Err(e) => {
            drop(engine);
            write_error(out, "400 Bad Request", "invalid_request_error", &e)
        }
    }
}

fn messages(
    request: &HttpRequest,
    out: &mut TcpStream,
    engine: &Mutex<Engine>,
    options: &Options,
) -> io::Result<()> {
    let Some(body) = body_of(request, out)? else {
        return Ok(());
    };
    // Held across generation: one model, one session, one request at a time.
    let mut engine = engine.lock().unwrap_or_else(|e| e.into_inner());
    let parsed = match engine.parse_request(&body, options) {
        Ok(parsed) => parsed,
        Err(e) => return write_error(out, "400 Bad Request", "invalid_request_error", &e),
    };
    let id = next_id("msg_");
    let started = Instant::now();
    let log = |engine: &Engine, outcome: &str| {
        if options.verbose {
            eprintln!(
                "kllm: {} {outcome} in {:.2} s | prompt {} reused + {} read | {} of {} context",
                if parsed.stream { "stream" } else { "message" },
                started.elapsed().as_secs_f64(),
                engine.reused,
                engine.fed,
                engine.session.position(),
                engine.context,
            );
        }
    };

    if parsed.stream {
        // The status line must go out before generation, so a prompt that is
        // rejected at the first forward pass can only be reported in-band.
        write!(
            out,
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/event-stream\r\n\
             Cache-Control: no-cache\r\n\
             Transfer-Encoding: chunked\r\n\
             Connection: keep-alive\r\n\r\n"
        )?;
        out.flush()?;
        let mut sink = Streamed {
            out,
            id: id.clone(),
            model: parsed.model.clone(),
            index: 0,
            open: None,
        };
        let result = engine.generate(&parsed, &mut |delta| sink.push(delta));
        let outcome = match result {
            Ok(()) => "ok",
            Err(Fail::Gone) => "client left",
            Err(Fail::Request(message)) | Err(Fail::Engine(message)) => {
                let _ = sink.event("error", &error_body("invalid_request_error", &message));
                "failed"
            }
        };
        // Terminate the chunked body so the connection stays reusable.
        let _ = sink.out.write_all(b"0\r\n\r\n");
        let _ = sink.out.flush();
        log(&engine, outcome);
        return Ok(());
    }

    let mut buffered = Buffered::default();
    match engine.generate(&parsed, &mut |delta| {
        buffered.push(delta);
        Ok(())
    }) {
        Ok(()) => {
            log(&engine, "ok");
            drop(engine);
            let response = buffered.finish(&id, &parsed.model);
            write_response(out, "200 OK", response.to_json().as_bytes())
        }
        Err(Fail::Gone) => Ok(()),
        Err(Fail::Request(message)) => {
            log(&engine, "rejected");
            drop(engine);
            write_error(out, "400 Bad Request", "invalid_request_error", &message)
        }
        Err(Fail::Engine(message)) => {
            log(&engine, "failed");
            drop(engine);
            write_error(out, "500 Internal Server Error", "api_error", &message)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn markers(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn holds_back_a_partial_marker_only() {
        let stop = markers(&["</done>"]);
        // A complete prefix of the marker must not be shown yet.
        assert_eq!(held_back("all good </d", &stop), 3);
        // Text that cannot grow into it is free to go.
        assert_eq!(held_back("all good", &stop), 0);
        // A whole marker is a match, not a hold.
        assert_eq!(held_back("all good </done>", &stop), 0);
    }

    #[test]
    fn holding_respects_character_boundaries() {
        let stop = markers(&["→stop"]);
        let held = held_back("wait →", &stop);
        assert_eq!(held, "→".len());
        // The split point the caller derives has to be a char boundary.
        assert!("wait →".is_char_boundary("wait →".len() - held));
    }

    #[test]
    fn finds_the_earliest_marker() {
        let list = markers(&["END", "STOP"]);
        assert_eq!(first_marker("a STOP b END", 0, &list), Some((2, 1)));
        // Searching past the first hit finds the later one.
        assert_eq!(first_marker("a STOP b END", 3, &list), Some((9, 0)));
        assert_eq!(first_marker("nothing here", 0, &list), None);
    }

    #[test]
    fn reads_a_qwen_tool_call() {
        let (name, input) =
            parse_tool_call("\n{\"name\": \"Read\", \"arguments\": {\"path\": \"a.txt\"}}\n")
                .unwrap();
        assert_eq!(name, "Read");
        assert_eq!(input, r#"{"path":"a.txt"}"#);
    }

    #[test]
    fn a_nameless_or_malformed_call_is_not_a_tool_call() {
        assert!(parse_tool_call("let me think about it").is_none());
        assert!(parse_tool_call(r#"{"arguments":{}}"#).is_none());
        assert!(parse_tool_call(r#"{"name":""}"#).is_none());
    }

    #[test]
    fn a_call_without_arguments_still_gets_an_object() {
        let (name, input) = parse_tool_call(r#"{"name":"ListDir"}"#).unwrap();
        assert_eq!(name, "ListDir");
        assert_eq!(input, "{}");
    }

    #[test]
    fn text_comes_from_either_content_shape() {
        assert_eq!(text_of(&string("hi")), "hi");
        let blocks =
            Value::parse(r#"[{"type":"text","text":"a"},{"type":"text","text":"b"}]"#).unwrap();
        assert_eq!(text_of(&blocks), "a\nb");
    }

    #[test]
    fn an_image_block_says_so_rather_than_vanishing() {
        let blocks = Value::parse(r#"[{"type":"image","source":{}}]"#).unwrap();
        assert!(text_of(&blocks).contains("cannot read images"));
    }

    #[test]
    fn common_prefix_counts_shared_leading_tokens() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 9, 4]), 2);
        assert_eq!(common_prefix(&[], &[1]), 0);
        assert_eq!(common_prefix(&[1, 2], &[1, 2]), 2);
    }

    #[test]
    fn buffered_text_deltas_join_into_one_block() {
        let mut buffered = Buffered::default();
        buffered.push(Delta::Prompt(7));
        buffered.push(Delta::Text("Hel"));
        buffered.push(Delta::Text("lo"));
        buffered.push(Delta::Done {
            stop_reason: "end_turn",
            stop_sequence: None,
            output_tokens: 2,
        });
        let json = buffered.finish("msg_1", "local").to_json();
        assert!(
            json.contains(r#""content":[{"type":"text","text":"Hello"}]"#),
            "{json}"
        );
        assert!(
            json.contains(r#""input_tokens":7,"output_tokens":2"#),
            "{json}"
        );
        assert!(json.contains(r#""stop_reason":"end_turn""#), "{json}");
    }

    #[test]
    fn a_tool_delta_starts_a_new_block_after_text() {
        let mut buffered = Buffered::default();
        buffered.push(Delta::Text("working on it"));
        buffered.push(Delta::Tool {
            id: "toolu_x".into(),
            name: "Read".into(),
            input: r#"{"path":"a"}"#.into(),
        });
        buffered.push(Delta::Text("done"));
        buffered.push(Delta::Done {
            stop_reason: "tool_use",
            stop_sequence: None,
            output_tokens: 9,
        });
        let json = buffered.finish("msg_1", "local").to_json();
        // Three blocks, and the tool input is an object rather than a string.
        assert!(
            json.contains(
                r#"{"type":"tool_use","id":"toolu_x","name":"Read","input":{"path":"a"}}"#
            ),
            "{json}"
        );
        assert!(json.contains(r#"{"type":"text","text":"done"}"#), "{json}");
    }

    #[test]
    fn tool_ids_do_not_repeat() {
        let a = next_id("toolu_");
        let b = next_id("toolu_");
        assert_ne!(a, b);
        assert!(a.starts_with("toolu_"));
    }
}
