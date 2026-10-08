#![allow(clippy::too_many_arguments, clippy::type_complexity)]
//! The bottom row of Fig. 3: the three reference panels redrawn in Rust with
//! plotters, from the data pyferrite read back. Geometry, colours, limits and
//! ticks follow `figures/style.json` and the matplotlib layout of the notebook.

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use serde_json::Value as Json;

pub const FONT: &str = "Times New Roman";

/// Everything the figure needs, already extracted from pyferrite values.
pub struct Data {
    pub t: Vec<f64>,
    pub sin: Vec<f64>,
    pub parabola: Vec<f64>,
    pub knn_x: Vec<(f64, f64)>,
    pub knn_y: Vec<i64>,
    pub knn_names: Vec<String>,
    pub field: Vec<Vec<f64>>, // rows: y, columns: x
}

pub struct Style {
    pub w: u32,
    pub h: u32,
    px_per_pt: f64,
    font_pt: f64,
    title_pt: f64,
    line_pt: f64,
    marker_pt: f64,
    grid: RGBAColor,
    colors: Vec<RGBColor>,
    viridis: Vec<RGBColor>,
    vmax: f64,
}

fn hex(s: &str) -> RGBColor {
    let v = u32::from_str_radix(s.trim_start_matches('#'), 16).unwrap_or(0);
    RGBColor((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

impl Style {
    pub fn from_json(j: &Json) -> Self {
        let dpi = j["dpi"].as_f64().unwrap_or(600.0);
        let g: Vec<f64> = j["grid_rgba"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        Style {
            w: (j["width_in"].as_f64().unwrap() * dpi).round() as u32,
            h: (j["height_in"].as_f64().unwrap() * dpi).round() as u32,
            px_per_pt: dpi / 72.0,
            font_pt: j["font_pt"].as_f64().unwrap(),
            title_pt: j["title_pt"].as_f64().unwrap(),
            line_pt: j["line_pt"].as_f64().unwrap(),
            marker_pt: j["marker_pt"].as_f64().unwrap(),
            grid: RGBAColor(
                (g[0] * 255.0) as u8,
                (g[1] * 255.0) as u8,
                (g[2] * 255.0) as u8,
                g[3],
            ),
            colors: j["colors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| hex(c.as_str().unwrap()))
                .collect(),
            viridis: j["viridis"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    let c: Vec<f64> = c
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap())
                        .collect();
                    RGBColor(
                        (c[0] * 255.0).round() as u8,
                        (c[1] * 255.0).round() as u8,
                        (c[2] * 255.0).round() as u8,
                    )
                })
                .collect(),
            vmax: j["field_vmax"].as_f64().unwrap_or(1.0),
        }
    }
    fn pt(&self, v: f64) -> f64 {
        v * self.px_per_pt
    }
    fn font(&self, pt: f64) -> TextStyle<'static> {
        (FONT, self.pt(pt)).into_font().color(&BLACK)
    }
    fn cmap(&self, v: f64) -> RGBColor {
        let k = ((v / self.vmax).clamp(0.0, 1.0) * 255.0).round() as usize;
        self.viridis[k]
    }
}

/// A matplotlib-like axes box in pixel coordinates.
struct Axes {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    xlim: (f64, f64),
    ylim: (f64, f64),
}

impl Axes {
    fn px(&self, x: f64, y: f64) -> (i32, i32) {
        let fx = (x - self.xlim.0) / (self.xlim.1 - self.xlim.0);
        let fy = (y - self.ylim.0) / (self.ylim.1 - self.ylim.0);
        (
            (self.x0 + fx * (self.x1 - self.x0)).round() as i32,
            (self.y1 - fy * (self.y1 - self.y0)).round() as i32,
        )
    }
}

/// Tick label with a typographic minus, as matplotlib prints it.
fn tick(v: f64, decimals: usize) -> String {
    let v = if v.abs() < 1e-12 { 0.0 } else { v };
    format!("{v:.decimals$}").replace('-', "\u{2212}")
}

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn frame<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    s: &Style,
    a: &Axes,
    xt: &[f64],
    yt: &[f64],
    dec: (usize, usize),
    labels: (&str, &str),
    title: &str,
    grid: bool,
) -> R<()>
where
    DB::ErrorType: 'static,
{
    let lw = s.pt(0.8).round() as u32;
    let tl = s.pt(3.5);
    let f = s.font(s.font_pt);
    // grid
    if grid {
        let gs = ShapeStyle::from(&s.grid).stroke_width(s.pt(0.6).round() as u32);
        for &x in xt {
            let (px, _) = a.px(x, 0.0);
            root.draw(&PathElement::new(
                vec![(px, a.y0 as i32), (px, a.y1 as i32)],
                gs,
            ))?;
        }
        for &y in yt {
            let (_, py) = a.px(0.0, y);
            root.draw(&PathElement::new(
                vec![(a.x0 as i32, py), (a.x1 as i32, py)],
                gs,
            ))?;
        }
    }
    // spines
    let r = [(a.x0 as i32, a.y0 as i32), (a.x1 as i32, a.y1 as i32)];
    root.draw(&Rectangle::new(
        r,
        ShapeStyle::from(&BLACK).stroke_width(lw),
    ))?;
    // ticks and tick labels
    let ts = ShapeStyle::from(&BLACK).stroke_width(lw);
    let gap = s.pt(3.5);
    for &x in xt {
        let (px, _) = a.px(x, 0.0);
        root.draw(&PathElement::new(
            vec![(px, a.y1 as i32), (px, (a.y1 + tl) as i32)],
            ts,
        ))?;
        root.draw(&Text::new(
            tick(x, dec.0),
            (px, (a.y1 + tl + gap) as i32),
            f.clone().pos(Pos::new(HPos::Center, VPos::Top)),
        ))?;
    }
    for &y in yt {
        let (_, py) = a.px(0.0, y);
        root.draw(&PathElement::new(
            vec![((a.x0 - tl) as i32, py), (a.x0 as i32, py)],
            ts,
        ))?;
        root.draw(&Text::new(
            tick(y, dec.1),
            ((a.x0 - tl - gap) as i32, py),
            f.clone().pos(Pos::new(HPos::Right, VPos::Center)),
        ))?;
    }
    // axis labels and title
    let xl_y = a.y1 + tl + gap + s.pt(s.font_pt) * 1.25;
    root.draw(&Text::new(
        labels.0.to_string(),
        (((a.x0 + a.x1) / 2.0) as i32, xl_y as i32),
        f.clone().pos(Pos::new(HPos::Center, VPos::Top)),
    ))?;
    let widest = yt
        .iter()
        .map(|y| tick(*y, dec.1).chars().count())
        .max()
        .unwrap_or(1) as f64;
    let yl_x = a.x0 - tl - gap - widest * s.pt(s.font_pt) * 0.5 - s.pt(4.0);
    root.draw(&Text::new(
        labels.1.to_string(),
        (yl_x as i32, ((a.y0 + a.y1) / 2.0) as i32),
        f.clone()
            .transform(FontTransform::Rotate270)
            .pos(Pos::new(HPos::Center, VPos::Bottom)),
    ))?;
    root.draw(&Text::new(
        title.to_string(),
        (((a.x0 + a.x1) / 2.0) as i32, (a.y0 - s.pt(6.0)) as i32),
        s.font(s.title_pt).pos(Pos::new(HPos::Center, VPos::Bottom)),
    ))?;
    Ok(())
}

/// Legend box, one row per entry or all entries in one row (`horizontal`, as
/// matplotlib's `ncol=len`); the closure in each entry draws its sample glyph.
fn legend<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    s: &Style,
    anchor: (f64, f64, HPos),
    horizontal: bool,
    entries: &[(
        String,
        RGBColor,
        &dyn Fn(&DrawingArea<DB, Shift>, (i32, i32), RGBColor) -> R<()>,
    )],
) -> R<()>
where
    DB::ErrorType: 'static,
{
    let fs = s.font_pt - 1.0;
    let f = s.font(fs);
    let row = s.pt(fs) * 1.2;
    let pad = s.pt(fs) * 0.4;
    let glyph = s.pt(fs) * 1.6;
    let gap = s.pt(fs) * 0.4;
    let col_gap = s.pt(fs) * 0.8;
    let text_w = |e: &str| e.chars().count() as f64 * s.pt(fs) * 0.42;
    let item_w: Vec<f64> = entries.iter().map(|e| glyph + gap + text_w(&e.0)).collect();
    let (w, h) = if horizontal {
        let sum: f64 = item_w.iter().sum();
        (
            pad * 2.0 + sum + col_gap * (entries.len() as f64 - 1.0),
            pad * 2.0 + row - s.pt(fs) * 0.2,
        )
    } else {
        (
            pad * 2.0 + item_w.iter().cloned().fold(0.0, f64::max),
            pad * 2.0 + row * entries.len() as f64 - s.pt(fs) * 0.2,
        )
    };
    let x0 = match anchor.2 {
        HPos::Center => anchor.0 - w / 2.0,
        _ => anchor.0,
    };
    let y0 = anchor.1;
    root.draw(&Rectangle::new(
        [(x0 as i32, y0 as i32), ((x0 + w) as i32, (y0 + h) as i32)],
        ShapeStyle::from(&WHITE.mix(0.95)).filled(),
    ))?;
    root.draw(&Rectangle::new(
        [(x0 as i32, y0 as i32), ((x0 + w) as i32, (y0 + h) as i32)],
        ShapeStyle::from(&RGBColor(204, 204, 204)).stroke_width(s.pt(0.8).round() as u32),
    ))?;
    let mut x = x0 + pad;
    for (i, (label, color, draw)) in entries.iter().enumerate() {
        let (ex, cy) = if horizontal {
            let ex = x;
            x += item_w[i] + col_gap;
            (ex, y0 + pad + row / 2.0 - s.pt(fs) * 0.1)
        } else {
            (
                x0 + pad,
                y0 + pad + row * i as f64 + row / 2.0 - s.pt(fs) * 0.1,
            )
        };
        draw(root, ((ex + glyph / 2.0) as i32, cy as i32), *color)?;
        root.draw(&Text::new(
            label.clone(),
            ((ex + glyph + gap) as i32, cy as i32),
            f.clone().pos(Pos::new(HPos::Left, VPos::Center)),
        ))?;
    }
    Ok(())
}

/// Matplotlib-style gridspec: left/right/bottom/top fractions, wspace, ratios.
fn layout(s: &Style) -> Vec<(f64, f64, f64, f64)> {
    let (w, h) = (s.w as f64, s.h as f64);
    let (left, right, bottom, top, wspace) = (0.075, 0.945, 0.185, 0.885, 0.34);
    let ratios = [1.0, 1.0, 1.12];
    let avail = (right - left) * w;
    let cell = avail / (3.0 + wspace * 2.0);
    let sum: f64 = ratios.iter().sum();
    let mut x = left * w;
    let mut out = Vec::new();
    for r in ratios {
        let cw = r * cell * 3.0 / sum;
        out.push((x, (1.0 - top) * h, x + cw, (1.0 - bottom) * h));
        x += cw + wspace * cell;
    }
    out
}

pub fn draw<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    s: &Style,
    d: &Data,
    tag: &str,
) -> R<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let cells = layout(s);
    let lw = s.pt(s.line_pt).round() as u32;
    let mr = (s.pt(s.marker_pt) / 2.0).round() as i32;

    // (a) time series
    let c = cells[0];
    let a = Axes {
        x0: c.0,
        y0: c.1,
        x1: c.2,
        y1: c.3,
        xlim: (0.0, 100.0),
        ylim: (-1.3, 1.9),
    };
    frame(
        &root,
        s,
        &a,
        &[0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
        &[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        (0, 1),
        ("t", "y"),
        &format!("{tag}: временной ряд (.h5)"),
        true,
    )?;
    let area = root.clone();
    for (k, ys) in [&d.sin, &d.parabola].iter().enumerate() {
        let col = s.colors[k];
        let pts: Vec<(i32, i32)> =
            d.t.iter()
                .zip(ys.iter())
                .map(|(t, y)| a.px(*t, *y))
                .collect();
        area.draw(&PathElement::new(
            pts.clone(),
            ShapeStyle::from(&col).stroke_width(lw),
        ))?;
        for p in pts.iter().step_by(5) {
            if k == 0 {
                area.draw(&Circle::new(*p, mr, ShapeStyle::from(&col).filled()))?;
            } else {
                area.draw(&Rectangle::new(
                    [(p.0 - mr, p.1 - mr), (p.0 + mr, p.1 + mr)],
                    ShapeStyle::from(&col).filled(),
                ))?;
            }
        }
    }
    let line_glyph = |circle: bool| {
        move |r: &DrawingArea<DB, Shift>, (x, y): (i32, i32), col: RGBColor| -> R<()> {
            let half = (s.pt(s.font_pt - 1.0) * 0.85) as i32;
            r.draw(&PathElement::new(
                vec![(x - half, y), (x + half, y)],
                ShapeStyle::from(&col).stroke_width(lw),
            ))?;
            if circle {
                r.draw(&Circle::new((x, y), mr, ShapeStyle::from(&col).filled()))?;
            } else {
                r.draw(&Rectangle::new(
                    [(x - mr, y - mr), (x + mr, y + mr)],
                    ShapeStyle::from(&col).filled(),
                ))?;
            }
            Ok(())
        }
    };
    let g_sin = line_glyph(true);
    let g_par = line_glyph(false);
    legend(
        &root,
        s,
        (
            (a.x0 + a.x1) / 2.0,
            a.y0 + s.pt(s.font_pt) * 0.5,
            HPos::Center,
        ),
        true,
        &[
            ("sin".to_string(), s.colors[0], &g_sin),
            ("парабола".to_string(), s.colors[1], &g_par),
        ],
    )?;

    // (b) k-NN training set
    let c = cells[1];
    let b = Axes {
        x0: c.0,
        y0: c.1,
        x1: c.2,
        y1: c.3,
        xlim: (-3.0, 4.0),
        ylim: (-3.0, 5.2),
    };
    frame(
        &root,
        s,
        &b,
        &(-3..=4).map(f64::from).collect::<Vec<_>>(),
        &(-3..=5).map(f64::from).collect::<Vec<_>>(),
        (0, 0),
        ("ГК 1", "ГК 2"),
        &format!("{tag}: выборка k-NN (.joblib)"),
        true,
    )?;
    let sr = (s.pt((s.marker_pt * s.marker_pt * 1.6).sqrt()) / 2.0).round() as i32;
    let edge = s.pt(0.4).round().max(1.0) as u32;
    for k in 0..3i64 {
        let col = s.colors[k as usize];
        for (p, y) in d.knn_x.iter().zip(&d.knn_y) {
            if *y == k {
                let q = b.px(p.0, p.1);
                root.draw(&Circle::new(q, sr, ShapeStyle::from(&col).filled()))?;
                root.draw(&Circle::new(
                    q,
                    sr,
                    ShapeStyle::from(&WHITE).stroke_width(edge),
                ))?;
            }
        }
    }
    let dot = |r: &DrawingArea<DB, Shift>, (x, y): (i32, i32), col: RGBColor| -> R<()> {
        r.draw(&Circle::new(
            (x, y),
            (sr as f64 * 0.9) as i32,
            ShapeStyle::from(&col).filled(),
        ))?;
        Ok(())
    };
    let entries: Vec<(
        String,
        RGBColor,
        &dyn Fn(&DrawingArea<DB, Shift>, (i32, i32), RGBColor) -> R<()>,
    )> = d
        .knn_names
        .iter()
        .enumerate()
        .map(|(k, n)| (n.clone(), s.colors[k], &dot as _))
        .collect();
    legend(
        &root,
        s,
        (
            b.x0 + s.pt(s.font_pt) * 0.5,
            b.y0 + s.pt(s.font_pt) * 0.5,
            HPos::Left,
        ),
        false,
        &entries,
    )?;

    // (c) heatmap of the torch tensor, with a colour bar
    let c = cells[2];
    let width = c.2 - c.0;
    let hx1 = c.0 + width * 0.90;
    let h = Axes {
        x0: c.0,
        y0: c.1,
        x1: hx1,
        y1: c.3,
        xlim: (-3.0, 3.0),
        ylim: (-3.0, 3.0),
    };
    let n_y = d.field.len();
    let n_x = d.field[0].len();
    for (j, row) in d.field.iter().enumerate() {
        for (i, v) in row.iter().enumerate() {
            // matplotlib imshow with extent: pixel centres span the extent edge to edge
            let xa = -3.0 + i as f64 * 6.0 / n_x as f64;
            let ya = -3.0 + j as f64 * 6.0 / n_y as f64;
            let p0 = h.px(xa, ya + 6.0 / n_y as f64);
            let p1 = h.px(xa + 6.0 / n_x as f64, ya);
            root.draw(&Rectangle::new(
                [p0, (p1.0 + 1, p1.1 + 1)],
                ShapeStyle::from(&s.cmap(*v)).filled(),
            ))?;
        }
    }
    frame(
        &root,
        s,
        &h,
        &(-3..=3).map(f64::from).collect::<Vec<_>>(),
        &(-3..=3).map(f64::from).collect::<Vec<_>>(),
        (0, 0),
        ("x", "y"),
        &format!("{tag}: тензор PyTorch (.pt)"),
        true,
    )?;
    // colour bar: matplotlib's default aspect of 20 inside the reserved strip
    let bar_h = c.3 - c.1;
    let bar_w = bar_h / 20.0;
    let bx0 = hx1 + width * 0.03;
    let steps = 256;
    for k in 0..steps {
        let v0 = s.vmax * k as f64 / steps as f64;
        let ya = c.3 - bar_h * (k as f64 / steps as f64);
        let yb = c.3 - bar_h * ((k + 1) as f64 / steps as f64);
        root.draw(&Rectangle::new(
            [
                (bx0 as i32, yb as i32),
                ((bx0 + bar_w) as i32, ya as i32 + 1),
            ],
            ShapeStyle::from(&s.cmap(v0 + s.vmax / steps as f64 / 2.0)).filled(),
        ))?;
    }
    let lw0 = s.pt(0.8).round() as u32;
    root.draw(&Rectangle::new(
        [(bx0 as i32, c.1 as i32), ((bx0 + bar_w) as i32, c.3 as i32)],
        ShapeStyle::from(&BLACK).stroke_width(lw0),
    ))?;
    for v in [0.0, 0.5, 1.0] {
        let y = c.3 - bar_h * v / s.vmax;
        root.draw(&PathElement::new(
            vec![
                ((bx0 + bar_w) as i32, y as i32),
                ((bx0 + bar_w + s.pt(3.5)) as i32, y as i32),
            ],
            ShapeStyle::from(&BLACK).stroke_width(lw0),
        ))?;
        root.draw(&Text::new(
            tick(v, 1),
            ((bx0 + bar_w + s.pt(7.0)) as i32, y as i32),
            s.font(s.font_pt).pos(Pos::new(HPos::Left, VPos::Center)),
        ))?;
    }
    root.present()?;
    Ok(())
}
