//! Read every file listed in `data/manifest.json` with pyferrite and compare
//! what comes back with what Python wrote: paths, types, shapes, checksums,
//! first and last elements, and exact scalar values.

use pyferrite::prelude::{read, read_with, Array, CastPolicy, DType, ReadOptions, Value};
use serde_json::{json, Value as Json};
use std::collections::BTreeMap;
use std::path::Path;

/// One flattened leaf of a value tree, keyed by its path.
#[derive(Debug)]
enum Leaf {
    Array {
        dtype: String,
        shape: Vec<usize>,
        sum: Option<f64>,
        first: Option<Json>,
        last: Option<f64>,
    },
    Seq {
        kind: &'static str,
        len: usize,
    },
    Set(Vec<String>),
    Bool(bool),
    Int(String),
    Float(f64),
    Complex(f64, f64),
    Str(String),
    Bytes(String),
    None,
}

fn dtype_name(d: &DType) -> String {
    match d {
        DType::Bool => "bool",
        DType::I8 => "int8",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U8 => "uint8",
        DType::U16 => "uint16",
        DType::U32 => "uint32",
        DType::U64 => "uint64",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::C64 => "complex64",
        DType::C128 => "complex128",
        DType::Str(_) => "str",
        DType::Bytes(_) => "bytes",
        other => return format!("{other:?}"),
    }
    .to_string()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Python-style `repr` for set members, matching `repr()` of ints and strings.
fn py_repr(v: &Value) -> String {
    match v {
        Value::Int(i) => i.to_string(),
        Value::Str(s) => format!("'{s}'"),
        Value::Float(f) => format!("{f:?}"),
        Value::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        other => format!("{other:?}"),
    }
}

/// Little-endian two's-complement bytes (pickle LONG) to a decimal string.
fn bigint(b: &[u8]) -> String {
    if b.len() <= 16 {
        let neg = b.last().map(|x| x & 0x80 != 0).unwrap_or(false);
        let mut buf = [if neg { 0xff } else { 0 }; 16];
        buf[..b.len()].copy_from_slice(b);
        return i128::from_le_bytes(buf).to_string();
    }
    format!("<{}-byte integer>", b.len())
}

fn summarize(a: &Array) -> Leaf {
    let dtype = dtype_name(&a.dtype());
    let shape = a.shape().to_vec();
    let (sum, first, last) = match a {
        Array::C64(x) => (
            Some(x.iter().map(|c| c.re as f64 + c.im as f64).sum()),
            None,
            None,
        ),
        Array::C128(x) => (Some(x.iter().map(|c| c.re + c.im).sum()), None, None),
        Array::Str(x) => (None, x.iter().next().map(|s| json!(s)), None),
        Array::Bytes(x) => (None, x.iter().next().map(|b| json!(hex(b))), None),
        other => match other.cast(&DType::F64, CastPolicy::Saturate) {
            Ok(Array::F64(f)) => {
                let v: Vec<f64> = f.iter().copied().collect();
                (
                    Some(v.iter().sum()),
                    v.first().map(|x| json!(x)),
                    v.last().copied(),
                )
            }
            _ => (None, None, None),
        },
    };
    Leaf::Array {
        dtype,
        shape,
        sum,
        first,
        last,
    }
}

fn join(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{path}/{key}")
    }
}

fn flatten(v: &Value, path: &str, out: &mut BTreeMap<String, Leaf>) {
    match v {
        Value::Dict(d) => {
            for (k, val) in d.iter() {
                let key = match k {
                    Value::Str(s) => s.clone(),
                    other => py_repr(other),
                };
                flatten(val, &join(path, &key), out);
            }
        }
        Value::List(items) | Value::Tuple(items) => {
            let kind = if matches!(v, Value::List(_)) {
                "list"
            } else {
                "tuple"
            };
            out.insert(
                path.to_string(),
                Leaf::Seq {
                    kind,
                    len: items.len(),
                },
            );
            for (i, it) in items.iter().enumerate() {
                flatten(it, &format!("{path}[{i}]"), out);
            }
        }
        Value::Set(items) => {
            let mut r: Vec<String> = items.iter().map(py_repr).collect();
            r.sort();
            out.insert(path.to_string(), Leaf::Set(r));
        }
        Value::Frame(f) => {
            for s in &f.columns {
                out.insert(join(path, &s.name), summarize(&s.values));
            }
        }
        // a pickled pandas.DataFrame: reassemble it into a Frame
        Value::Object(o) if o.module.starts_with("pandas") => {
            if let Ok(Some(f)) = pyferrite::interop::pandas::to_frame(v) {
                flatten(&Value::Frame(f), path, out);
            }
        }
        Value::Object(o) => {
            // an inert sklearn object: its fitted attributes live in `state`
            if let Some(state) = &o.state {
                flatten(state, path, out);
            }
        }
        Value::Array(a) => {
            out.insert(path.to_string(), summarize(a));
        }
        Value::Bool(b) => {
            out.insert(path.to_string(), Leaf::Bool(*b));
        }
        Value::Int(i) => {
            out.insert(path.to_string(), Leaf::Int(i.to_string()));
        }
        Value::BigInt(b) => {
            out.insert(path.to_string(), Leaf::Int(bigint(b)));
        }
        Value::Float(f) => {
            out.insert(path.to_string(), Leaf::Float(*f));
        }
        Value::Complex(c) => {
            out.insert(path.to_string(), Leaf::Complex(c.re, c.im));
        }
        Value::Str(s) => {
            out.insert(path.to_string(), Leaf::Str(s.clone()));
        }
        Value::Bytes(b) => {
            out.insert(path.to_string(), Leaf::Bytes(hex(b)));
        }
        Value::None => {
            out.insert(path.to_string(), Leaf::None);
        }
    }
}

fn close(a: f64, b: f64, scale: f64) -> bool {
    a == b || (a - b).abs() <= 1e-9 * scale.max(1.0)
}

/// Compare one manifest entry with the flattened value; `Err` explains a mismatch.
fn compare(e: &Json, got: Option<&Leaf>) -> std::result::Result<(), String> {
    let got = got.ok_or("missing")?;
    let kind = e["kind"].as_str().unwrap_or("");
    match (kind, got) {
        (
            "array",
            Leaf::Array {
                dtype,
                shape,
                sum,
                first,
                last,
            },
        ) => {
            if e["dtype"] != json!(dtype) {
                return Err(format!("dtype {dtype} != {}", e["dtype"]));
            }
            if e["shape"] != json!(shape) {
                return Err(format!("shape {shape:?} != {}", e["shape"]));
            }
            if let (Some(want), Some(have)) = (e["sum"].as_f64(), sum) {
                let scale =
                    e["sum"].as_f64().unwrap_or(0.0).abs() + shape.iter().product::<usize>() as f64;
                if !close(want, *have, scale) {
                    return Err(format!("sum {have} != {want}"));
                }
            }
            if !e["first"].is_null() && first.as_ref() != Some(&e["first"]) {
                return Err(format!("first {first:?} != {}", e["first"]));
            }
            if let (Some(want), Some(have)) = (e["last"].as_f64(), last) {
                if want != *have {
                    return Err(format!("last {have} != {want}"));
                }
            }
            Ok(())
        }
        (k @ ("list" | "tuple"), Leaf::Seq { kind, len }) if k == *kind => {
            if e["len"] == json!(len) {
                Ok(())
            } else {
                Err(format!("len {len} != {}", e["len"]))
            }
        }
        ("set", Leaf::Set(items)) => {
            if e["items"] == json!(items) {
                Ok(())
            } else {
                Err(format!("{items:?} != {}", e["items"]))
            }
        }
        ("bool", Leaf::Bool(b)) if e["value"] == json!(b) => Ok(()),
        ("int", Leaf::Int(s)) if e["value"] == json!(s) => Ok(()),
        ("float", Leaf::Float(f)) => {
            let want: f64 = e["value"]
                .as_str()
                .unwrap_or("")
                .parse()
                .map_err(|_| "bad float")?;
            if want.to_bits() == f.to_bits() {
                Ok(())
            } else {
                Err(format!("{f} != {want}"))
            }
        }
        ("complex", Leaf::Complex(re, im)) if e["re"] == json!(re) && e["im"] == json!(im) => {
            Ok(())
        }
        ("str", Leaf::Str(s)) if e["value"] == json!(s) => Ok(()),
        ("bytes", Leaf::Bytes(h)) if e["hex"] == json!(h) => Ok(()),
        ("none", Leaf::None) => Ok(()),
        (_, other) => Err(format!("expected {kind}, got {other:?}")),
    }
}

/// Check every file. Returns the JSON report and the number of failures.
pub fn run(data: &Path) -> std::result::Result<(Json, usize), Box<dyn std::error::Error>> {
    let manifest: Json =
        serde_json::from_str(&std::fs::read_to_string(data.join("manifest.json"))?)?;
    let mut files = Vec::new();
    let (mut ok_total, mut bad_total) = (0usize, 0usize);
    let mut by_ext: BTreeMap<String, (usize, usize)> = BTreeMap::new();

    for m in manifest.as_array().into_iter().flatten() {
        let name = m["file"].as_str().unwrap_or_default();
        let path = data.join(name);
        let ext = name.rsplit('.').next().unwrap_or("").to_string();
        let mut checks = Vec::new();

        let got: std::result::Result<Value, String> = (|| {
            Ok(if m["must_refuse"] == json!(true) {
                // A poisoned pickle (os.system): the default read must refuse it,
                // and even the permissive read must only describe the call.
                let refused = read(&path).is_err();
                checks.push(json!({"path": "(default read refuses os.system)", "ok": refused}));
                let v = read_with(&path, &ReadOptions::new().allow_unknown_globals(true))
                    .map_err(|e| e.to_string())?;
                let inert = matches!(v.as_dict().and_then(|d| d.get("hook")),
                                 Some(Value::Object(o)) if o.name == "system");
                checks
                    .push(json!({"path": "(permissive read keeps the call as data)", "ok": inert}));
                v
            } else if let Some(class) = m["object"].as_str() {
                // An arbitrary class is captured inertly by the default read.
                let v = read(&path).map_err(|e| e.to_string())?;
                let got = match &v {
                    Value::Object(o) => format!("{}.{}", o.module, o.name),
                    other => other.type_name().to_string(),
                };
                checks.push(
                    json!({"path": "(class captured, not run)", "ok": got == class, "got": got}),
                );
                v
            } else {
                read(&path).map_err(|e| e.to_string())?
            })
        })();
        let value = match got {
            Ok(v) => v,
            Err(why) => {
                checks.push(json!({"path": "(read)", "ok": false, "why": why}));
                Value::None
            }
        };
        let mut leaves = BTreeMap::new();
        flatten(&value, "", &mut leaves);
        for e in m["entries"].as_array().into_iter().flatten() {
            let p = e["path"].as_str().unwrap_or_default();
            match compare(e, leaves.get(p)) {
                Ok(()) => checks.push(json!({"path": p, "ok": true})),
                Err(why) => checks.push(json!({"path": p, "ok": false, "why": why})),
            }
        }
        let ok = checks.iter().filter(|c| c["ok"] == json!(true)).count();
        let bad = checks.len() - ok;
        ok_total += ok;
        bad_total += bad;
        let slot = by_ext.entry(ext).or_default();
        slot.0 += 1;
        slot.1 += ok;
        println!(
            "  {:<4} {:<32} {:>3}/{:<3} {}",
            if bad == 0 { "ok" } else { "FAIL" },
            name,
            ok,
            checks.len(),
            m["structure"].as_str().unwrap_or("")
        );
        for c in checks.iter().filter(|c| c["ok"] != json!(true)) {
            println!("         {} : {}", c["path"], c["why"]);
        }
        files.push(json!({"file": name, "passed": ok, "failed": bad, "checks": checks}));
    }
    let report = json!({
        "pyferrite": format!("{} (crates.io)", crate::PYFERRITE),
        "files": files.len(),
        "checks_passed": ok_total,
        "checks_failed": bad_total,
        "by_extension": by_ext.iter().map(|(k, v)| (k.clone(), json!({"files": v.0, "checks_passed": v.1}))).collect::<serde_json::Map<_, _>>(),
        "details": files,
    });
    Ok((report, bad_total))
}
