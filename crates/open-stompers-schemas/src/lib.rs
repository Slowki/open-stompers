//! FlatBuffers types generated from `schemas/`.
//!
//! Regenerate with `tools/generate-flatbuffers`.

#[allow(
    clippy::all,
    dead_code,
    deprecated,
    non_camel_case_types,
    non_upper_case_globals,
    unused,
    unused_imports
)]
#[path = "generated/mod.rs"]
mod generated;

pub use generated::open_stompers;
pub use open_stompers::RobotPosture;
