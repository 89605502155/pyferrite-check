//! pyferrite-check: read every file written by `validation.ipynb` with the
//! published pyferrite crate, check it against `data/manifest.json`, then draw
//! the three reference panels in Rust (bottom row of Fig. 3 of the article).
//!
//!     cargo run --release               # from the `rust/` directory
//!     cargo run --release -- ../data ../figures

mod check;
mod plot;

use plot::{Data, Style};
use plotters::prelude::*;
use pyferrite::prelude::*;
use std::path::{Path, PathBuf};

type R<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The pyferrite release this check was written against (see Cargo.toml).
pub const PYFERRITE: &str = "0.0.2";

fn floats(v: &Value, key: &str) -> R<Vec<f64>> {
    let a = v
        .as_dict()
        .and_then(|d| d.get(key))
        .and_then(|x| x.as_array())
        .ok_or(format!("no array `{key}`"))?;
    match a.cast(&DType::F64, CastPolicy::Strict)? {
        Array::F64(x) => Ok(x.iter().copied().collect()),
        _ => unreachable!(),
    }
}

fn load(data: &Path) -> R<Data> {
    let series = read(data.join("series.h5"))?;
    let knn = read(data.join("knn_params.joblib"))?;
    let field = read(data.join("field.pt"))?;

    let xy = floats(&knn, "X")?;
    let y = match knn
        .as_dict()
        .and_then(|d| d.get("y"))
        .and_then(|x| x.as_array())
    {
        Some(Array::I64(a)) => a.iter().copied().collect(),
        _ => return Err("knn `y` is not int64".into()),
    };
    let names = match knn
        .as_dict()
        .and_then(|d| d.get("classes"))
        .and_then(|x| x.as_array())
    {
        Some(Array::Str(a)) => a.iter().cloned().collect(),
        _ => return Err("knn `classes` is not a string array".into()),
    };
    let f = field
        .as_dict()
        .and_then(|d| d.get("field"))
        .and_then(|x| x.as_array())
        .ok_or("no field")?;
    let (ny, nx) = (f.shape()[0], f.shape()[1]);
    let flat = match f.cast(&DType::F64, CastPolicy::Strict)? {
        Array::F64(a) => a.iter().copied().collect::<Vec<f64>>(),
        _ => unreachable!(),
    };
    Ok(Data {
        t: floats(&series, "t")?,
        sin: floats(&series, "sin")?,
        parabola: floats(&series, "parabola")?,
        knn_x: xy.chunks_exact(2).map(|p| (p[0], p[1])).collect(),
        knn_y: y,
        knn_names: names,
        field: flat
            .chunks_exact(nx)
            .map(|r| r.to_vec())
            .collect::<Vec<_>>()
            .into_iter()
            .take(ny)
            .collect(),
    })
}

fn register_fonts() -> R<()> {
    let candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "C:\\Windows\\Fonts\\times.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ];
    let path = candidates.iter().find(|p| Path::new(p).exists()).ok_or(
        "Times New Roman not found; install it (e.g. ttf-mscorefonts-installer) or edit register_fonts()",
    )?;
    let bytes: &'static [u8] = Box::leak(std::fs::read(path)?.into_boxed_slice());
    plotters::style::register_font(plot::FONT, FontStyle::Normal, bytes)
        .map_err(|_| "bad font file")?;
    Ok(())
}

fn save_png(path: &Path, buf: &[u8], w: u32, h: u32, dpi: f64) -> R<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let ppm = (dpi / 0.0254).round() as u32;
    enc.set_pixel_dims(Some(png::PixelDimensions {
        xppu: ppm,
        yppu: ppm,
        unit: png::Unit::Meter,
    }));
    enc.write_header()?.write_image_data(buf)?;
    Ok(())
}

fn save_tiff(path: &Path, buf: &[u8], w: u32, h: u32, dpi: u32) -> R<()> {
    use tiff::encoder::{colortype::RGB8, compression::Lzw, Rational, TiffEncoder};
    use tiff::tags::ResolutionUnit;
    let mut enc = TiffEncoder::new(std::io::BufWriter::new(std::fs::File::create(path)?))?;
    let mut img = enc.new_image_with_compression::<RGB8, _>(w, h, Lzw)?;
    img.resolution(ResolutionUnit::Inch, Rational { n: dpi, d: 1 });
    img.write_data(buf)?;
    Ok(())
}

/// Decode an 8-bit RGB or RGBA PNG into RGB rows.
fn load_png_rgb(path: &Path) -> R<(Vec<u8>, u32, u32)> {
    let mut dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info()?;
    let mut raw = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut raw)?;
    let px = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        c => return Err(format!("{}: unsupported PNG colour type {c:?}", path.display()).into()),
    };
    let rgb = raw[..info.buffer_size()]
        .chunks_exact(px)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    Ok((rgb, info.width, info.height))
}

/// Fig. 3 of the article: the Python row drawn by the notebook on top, the
/// Rust row drawn here below it, so the two can be compared panel by panel.
fn compose(figs: &Path, rust: &[u8], w: u32, h: u32) -> R<()> {
    let (top, tw, th) = load_png_rgb(&figs.join("python_panels.png"))?;
    if tw != w {
        return Err(format!("python_panels.png is {tw} px wide, the Rust row is {w} px").into());
    }
    let mut buf = top;
    buf.extend_from_slice(rust);
    let hh = th + h;
    save_png(&figs.join("validation_figure.png"), &buf, w, hh, 600.0)?;
    save_tiff(&figs.join("validation_figure.tif"), &buf, w, hh, 600)?;
    Ok(())
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("../data"));
    let figs = PathBuf::from(args.get(2).map(String::as_str).unwrap_or("../figures"));

    println!(
        "pyferrite-check: reading {} with pyferrite {PYFERRITE} from crates.io\n",
        data.display()
    );
    let (report, failed) = check::run(&data)?;
    std::fs::write(
        figs.join("rust_report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    println!(
        "\n{} files, {} checks passed, {} failed\n",
        report["files"], report["checks_passed"], failed
    );

    register_fonts()?;
    let style: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(figs.join("style.json"))?)?;
    let s = Style::from_json(&style);
    let d = load(&data)?;

    let mut buf = vec![0u8; (s.w * s.h * 3) as usize];
    plot::draw(
        BitMapBackend::with_buffer(&mut buf, (s.w, s.h)).into_drawing_area(),
        &s,
        &d,
        "Rust",
    )?;
    save_png(&figs.join("rust_panels.png"), &buf, s.w, s.h, 600.0)?;
    save_tiff(&figs.join("rust_panels.tif"), &buf, s.w, s.h, 600)?;
    plot::draw(
        SVGBackend::new(&figs.join("rust_panels.svg"), (s.w, s.h)).into_drawing_area(),
        &s,
        &d,
        "Rust",
    )?;
    compose(&figs, &buf, s.w, s.h)?;
    println!("figures: rust_panels.png, rust_panels.tif (600 dpi), rust_panels.svg");
    println!("combined: validation_figure.png, validation_figure.tif (Python above, Rust below)");
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
