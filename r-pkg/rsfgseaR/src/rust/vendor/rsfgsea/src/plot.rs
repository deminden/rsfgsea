use crate::algo_support::{build_gene_index, extract_pathway_hits};
use crate::core::{Pathway, RankedList, ScoreType};
use ab_glyph::FontArc;
use anyhow::{Result, bail};
use image::imageops::rotate270;
use image::{Rgba, RgbaImage};
use imageproc::drawing::{
    draw_antialiased_line_segment_mut, draw_filled_rect_mut, draw_line_segment_mut, draw_text_mut,
    text_size,
};
use imageproc::pixelops::interpolate;
use imageproc::rect::Rect;
use png::{BitDepth, ColorType, Encoder, PixelDimensions, Unit};
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::OnceLock;

const WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);
const TRANSPARENT: Rgba<u8> = Rgba([255, 255, 255, 0]);
const BLACK: Rgba<u8> = Rgba([0, 0, 0, 255]);
const DARK_GRAY: Rgba<u8> = Rgba([70, 70, 70, 255]);
const LIGHT_GRAY: Rgba<u8> = Rgba([230, 230, 230, 255]);
const GREEN: Rgba<u8> = Rgba([0, 180, 0, 255]);
const RED: Rgba<u8> = Rgba([220, 40, 40, 255]);
const GRID_MIN_GRAY: u8 = 160;
const GRID_MAX_GRAY: u8 = 230;
const OPPOSITE_SIGN_AXIS_CLIP_RATIO: f64 = 0.05;

static PLOT_FONT: OnceLock<Result<FontArc, String>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct EnrichmentPlotOptions {
    pub width_inches: f64,
    pub height_inches: f64,
    pub dpi: u32,
    pub transparent_background: bool,
    pub title: Option<String>,
}

impl Default for EnrichmentPlotOptions {
    fn default() -> Self {
        Self {
            width_inches: 4.5,
            height_inches: 3.2,
            dpi: 300,
            transparent_background: false,
            title: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnrichmentPlotData {
    pub pathway_name: String,
    pub hit_positions: Vec<usize>,
    pub running_score: Vec<(usize, f64)>,
    pub es: f64,
}

pub fn build_enrichment_plot_data(
    ranks: &RankedList,
    pathway: &Pathway,
    score_type: ScoreType,
    gsea_param: f64,
) -> Result<EnrichmentPlotData> {
    if ranks.is_empty() {
        bail!("ranks must not be empty");
    }
    if gsea_param < 0.0 || !gsea_param.is_finite() {
        bail!("gsea_param must be a finite value greater than or equal to 0");
    }

    let gene_to_idx = build_gene_index(ranks);
    let hits = extract_pathway_hits(pathway, &gene_to_idx);
    if hits.is_empty() {
        bail!(
            "pathway '{}' has no genes present in the ranked list",
            pathway.name
        );
    }
    if hits.len() == ranks.len() {
        bail!(
            "pathway '{}' covers the full ranked list and cannot produce an enrichment curve",
            pathway.name
        );
    }

    let n_total = ranks.len();
    let n_hits = hits.len();
    let miss_penalty = 1.0 / (n_total - n_hits) as f64;
    let mut hit_cursor = 0usize;
    let mut hit_weights = Vec::with_capacity(n_hits);
    let mut norm = 0.0;
    for &idx in &hits {
        let weight = ranks.scores[idx].abs().powf(gsea_param);
        hit_weights.push(weight);
        norm += weight;
    }
    if norm == 0.0 {
        norm = n_hits as f64;
        hit_weights.fill(1.0);
    }

    let mut running = 0.0;
    let mut es_pos = f64::NEG_INFINITY;
    let mut es_neg = f64::INFINITY;
    let mut points = Vec::with_capacity(n_total + 1);
    points.push((0usize, 0.0));

    for i in 0..n_total {
        if hit_cursor < n_hits && hits[hit_cursor] == i {
            running += hit_weights[hit_cursor] / norm;
            hit_cursor += 1;
        } else {
            running -= miss_penalty;
        }
        es_pos = es_pos.max(running);
        es_neg = es_neg.min(running);
        points.push((i + 1, running));
    }

    let es = match score_type {
        ScoreType::Pos => es_pos,
        ScoreType::Neg => es_neg,
        ScoreType::Std => {
            if es_pos > -es_neg {
                es_pos
            } else if es_pos < -es_neg {
                es_neg
            } else {
                0.0
            }
        }
    };

    Ok(EnrichmentPlotData {
        pathway_name: pathway.name.clone(),
        hit_positions: hits.into_iter().map(|idx| idx + 1).collect(),
        running_score: points,
        es,
    })
}

pub fn write_enrichment_plot_png<P: AsRef<Path>>(
    ranks: &RankedList,
    pathway: &Pathway,
    output_path: P,
    score_type: ScoreType,
    gsea_param: f64,
    options: &EnrichmentPlotOptions,
) -> Result<()> {
    if !options.width_inches.is_finite()
        || !options.height_inches.is_finite()
        || options.width_inches <= 0.0
        || options.height_inches <= 0.0
        || options.dpi == 0
    {
        bail!("plot size in inches and dpi must be finite positive values");
    }

    let pixel_width = inches_to_pixels(options.width_inches, options.dpi);
    let pixel_height = inches_to_pixels(options.height_inches, options.dpi);
    if pixel_width < 320 || pixel_height < 240 {
        bail!("plot dimensions must be at least 320x240 pixels");
    }
    let dpi_scale = options.dpi as f32 / 300.0;

    let plot = build_enrichment_plot_data(ranks, pathway, score_type, gsea_param)?;
    let x_end = ranks.len().max(1);

    let mut data_y_min = plot
        .running_score
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min)
        .min(0.0)
        .min(plot.es);
    let mut data_y_max = plot
        .running_score
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0)
        .max(plot.es);
    if !data_y_min.is_finite() || !data_y_max.is_finite() {
        bail!("failed to compute finite plot bounds");
    }
    if (data_y_max - data_y_min).abs() < 1e-12 {
        data_y_min -= 0.1;
        data_y_max += 0.1;
    }
    let (y_min, y_max, y_step, y_ticks) = compute_y_axis(data_y_min, data_y_max, 6);

    let background = if options.transparent_background {
        TRANSPARENT
    } else {
        WHITE
    };
    let mut img = RgbaImage::from_pixel(pixel_width, pixel_height, background);

    let left_margin = scaled_px_i32(140.0, dpi_scale);
    let right_margin = scaled_px_i32(30.0, dpi_scale);
    let top_margin = scaled_px_i32(90.0, dpi_scale);
    let bottom_margin = scaled_px_i32(80.0, dpi_scale);
    let plot_left = left_margin;
    let plot_top = top_margin;
    let plot_width = pixel_width as i32 - left_margin - right_margin;
    let plot_height = pixel_height as i32 - top_margin - bottom_margin;
    if plot_width <= 0 || plot_height <= 0 {
        bail!("plot dimensions leave no drawable area");
    }
    let plot_rect = Rect::at(plot_left, plot_top).of_size(plot_width as u32, plot_height as u32);

    draw_filled_rect_mut(&mut img, plot_rect, background);
    draw_line_segment_mut(
        &mut img,
        (plot_left as f32, plot_top as f32),
        (plot_left as f32, (plot_top + plot_height) as f32),
        BLACK,
    );
    draw_line_segment_mut(
        &mut img,
        (plot_left as f32, (plot_top + plot_height) as f32),
        (
            (plot_left + plot_width) as f32,
            (plot_top + plot_height) as f32,
        ),
        BLACK,
    );
    draw_line_segment_mut(
        &mut img,
        (plot_left as f32, plot_top as f32),
        ((plot_left + plot_width) as f32, plot_top as f32),
        LIGHT_GRAY,
    );
    draw_line_segment_mut(
        &mut img,
        ((plot_left + plot_width) as f32, plot_top as f32),
        (
            (plot_left + plot_width) as f32,
            (plot_top + plot_height) as f32,
        ),
        LIGHT_GRAY,
    );

    let map_x =
        |x: usize| -> f32 { plot_left as f32 + (x as f32 / x_end as f32) * plot_width as f32 };
    let map_y = |y: f64| -> f32 {
        plot_top as f32 + ((y_max - y) / (y_max - y_min)) as f32 * plot_height as f32
    };

    let font = load_plot_font().map_err(|err| anyhow::anyhow!(err.clone()))?;
    let tick_scale = 26.0f32 * dpi_scale;
    let title_scale = 28.0f32 * dpi_scale;
    let axis_label_scale = 28.0f32 * dpi_scale;
    let grid_color = grid_color_for_dpi(options.dpi);
    let x_ticks = [0usize, x_end / 4, x_end / 2, (x_end * 3) / 4, x_end];
    let y_tick_labels: Vec<String> = y_ticks
        .iter()
        .map(|value| format_axis_tick(*value, y_step))
        .collect();
    let max_y_tick_label_width = y_tick_labels
        .iter()
        .map(|label| text_size(tick_scale, &font, label).0 as i32)
        .max()
        .unwrap_or(0);

    for value in x_ticks {
        let x = map_x(value);
        draw_thick_line_segment(
            &mut img,
            (x, plot_top as f32),
            (x, (plot_top + plot_height) as f32),
            scaled_px_i32(0.0, dpi_scale),
            grid_color,
        );
    }
    for &value in &y_ticks {
        if value.abs() < 1e-12 {
            continue;
        }
        let y = map_y(value);
        draw_thick_line_segment(
            &mut img,
            (plot_left as f32, y),
            ((plot_left + plot_width) as f32, y),
            scaled_px_i32(0.0, dpi_scale),
            grid_color,
        );
    }

    let zero_y = map_y(0.0);
    draw_line_segment_mut(
        &mut img,
        (plot_left as f32, zero_y),
        ((plot_left + plot_width) as f32, zero_y),
        BLACK,
    );
    let es_y = map_y(plot.es);
    draw_dotted_horizontal_line(
        &mut img,
        plot_left as f32,
        (plot_left + plot_width) as f32,
        es_y,
        scaled_px_i32(1.0, dpi_scale),
        12.0 * dpi_scale,
        8.0 * dpi_scale,
        RED,
    );

    for window in plot.running_score.windows(2) {
        let (x0, y0) = window[0];
        let (x1, y1) = window[1];
        draw_thick_line_segment(
            &mut img,
            (map_x(x0), map_y(y0)),
            (map_x(x1), map_y(y1)),
            scaled_px_i32(2.0, dpi_scale),
            GREEN,
        );
    }

    let rug_center = (plot_top + plot_height) as f32;
    let rug_bottom = rug_center + 18.0 * dpi_scale;
    let rug_top = rug_center - 24.0 * dpi_scale;
    let rug_width = scaled_px_i32(2.0, dpi_scale).max(1);
    for &hit in &plot.hit_positions {
        let x = map_x(hit);
        let rect = Rect::at(x.round() as i32 - rug_width / 2, rug_top.round() as i32).of_size(
            rug_width as u32,
            (rug_bottom - rug_top).round().max(1.0) as u32,
        );
        draw_filled_rect_mut(&mut img, rect, DARK_GRAY);
    }

    for (value, label) in y_ticks.iter().zip(y_tick_labels.iter()) {
        let y = map_y(*value);
        draw_line_segment_mut(
            &mut img,
            ((plot_left - scaled_px_i32(7.0, dpi_scale)) as f32, y),
            (plot_left as f32, y),
            BLACK,
        );
        let (label_width, _) = text_size(tick_scale, &font, label);
        draw_text(
            &mut img,
            &font,
            plot_left - scaled_px_i32(12.0, dpi_scale) - label_width as i32,
            y.round() as i32 - scaled_px_i32(12.0, dpi_scale),
            tick_scale,
            label,
            BLACK,
        );
    }

    for value in x_ticks {
        let x = map_x(value);
        draw_line_segment_mut(
            &mut img,
            (x, (plot_top + plot_height) as f32),
            (x, (plot_top + plot_height + 5) as f32),
            BLACK,
        );
        draw_text(
            &mut img,
            &font,
            x.round() as i32 - scaled_px_i32(12.0, dpi_scale),
            plot_top + plot_height + scaled_px_i32(24.0, dpi_scale),
            tick_scale,
            &value.to_string(),
            BLACK,
        );
    }

    if let Some(title) = options.title.as_deref() {
        draw_text_centered(
            &mut img,
            &font,
            pixel_width as i32 / 2,
            scaled_px_i32(24.0, dpi_scale),
            title_scale,
            title,
            BLACK,
        );
    }
    draw_text_centered(
        &mut img,
        &font,
        pixel_width as i32 / 2,
        pixel_height as i32 - scaled_px_i32(28.0, dpi_scale),
        axis_label_scale,
        "rank",
        BLACK,
    );
    draw_vertical_text_centered(
        &mut img,
        &font,
        plot_left - scaled_px_i32(30.0, dpi_scale) - max_y_tick_label_width,
        plot_top + plot_height / 2,
        axis_label_scale,
        "enrichment score",
        BLACK,
    );

    save_png_with_dpi(&img, output_path, options.dpi)?;
    Ok(())
}

fn save_png_with_dpi<P: AsRef<Path>>(img: &RgbaImage, output_path: P, dpi: u32) -> Result<()> {
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    let mut encoder = Encoder::new(writer, img.width(), img.height());
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    encoder.set_pixel_dims(Some(PixelDimensions {
        xppu: dpi_to_pixels_per_meter(dpi),
        yppu: dpi_to_pixels_per_meter(dpi),
        unit: Unit::Meter,
    }));

    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(img.as_raw())?;
    Ok(())
}

fn dpi_to_pixels_per_meter(dpi: u32) -> u32 {
    ((dpi as f64 / 0.0254).round() as u32).max(1)
}

fn inches_to_pixels(inches: f64, dpi: u32) -> u32 {
    ((inches * dpi as f64).round() as u32).max(1)
}

fn scaled_px_i32(px_at_300dpi: f32, dpi_scale: f32) -> i32 {
    (px_at_300dpi * dpi_scale).round() as i32
}

fn grid_color_for_dpi(dpi: u32) -> Rgba<u8> {
    let t = ((dpi as f32 - 300.0) / (1200.0 - 300.0)).clamp(0.0, 1.0);
    let gray =
        (GRID_MAX_GRAY as f32 + (GRID_MIN_GRAY as f32 - GRID_MAX_GRAY as f32) * t).round() as u8;
    Rgba([gray, gray, gray, 255])
}

fn load_plot_font() -> std::result::Result<&'static FontArc, &'static String> {
    let result = PLOT_FONT.get_or_init(|| {
        let candidates = [
            std::env::var("RSFGSEA_PLOT_FONT").ok(),
            Some("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf".to_string()),
            Some("/usr/share/fonts/fonts-go/Go-Regular.ttf".to_string()),
            Some("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf".to_string()),
        ];

        for path in candidates.into_iter().flatten() {
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            if let Ok(font) = FontArc::try_from_vec(bytes) {
                return Ok(font);
            }
        }

        Err(
            "failed to load a plotting font; set RSFGSEA_PLOT_FONT to a readable .ttf/.otf file"
                .to_string(),
        )
    });

    match result {
        Ok(font) => Ok(font),
        Err(err) => Err(err),
    }
}

fn draw_text(
    img: &mut RgbaImage,
    font: &FontArc,
    x: i32,
    y: i32,
    scale: f32,
    text: &str,
    color: Rgba<u8>,
) {
    draw_text_mut(img, color, x, y, scale, font, text);
}

fn draw_text_centered(
    img: &mut RgbaImage,
    font: &FontArc,
    center_x: i32,
    y: i32,
    scale: f32,
    text: &str,
    color: Rgba<u8>,
) {
    let (width, _) = text_size(scale, font, text);
    let start_x = center_x - width as i32 / 2;
    draw_text_mut(img, color, start_x, y, scale, font, text);
}

fn compute_y_axis(data_min: f64, data_max: f64, target_ticks: usize) -> (f64, f64, f64, Vec<f64>) {
    if data_min >= 0.0
        || (data_max > 0.0
            && data_min < 0.0
            && (-data_min / data_max) <= OPPOSITE_SIGN_AXIS_CLIP_RATIO)
    {
        let range = nice_number(data_max.max(1e-12), false);
        let step = nice_number(range / (target_ticks.saturating_sub(1).max(1) as f64), true);
        let axis_min = 0.0;
        let axis_max = (data_max / step).ceil() * step;

        let mut ticks = Vec::new();
        let mut value = axis_min;
        while value <= axis_max + step * 0.5 {
            ticks.push(if value.abs() < step * 1e-6 {
                0.0
            } else {
                value
            });
            value += step;
        }

        return (axis_min, axis_max, step, ticks);
    }

    if data_max <= 0.0
        || (data_min < 0.0
            && data_max > 0.0
            && (data_max / -data_min) <= OPPOSITE_SIGN_AXIS_CLIP_RATIO)
    {
        let range = nice_number((-data_min).max(1e-12), false);
        let step = nice_number(range / (target_ticks.saturating_sub(1).max(1) as f64), true);
        let axis_min = (data_min / step).floor() * step;
        let axis_max = 0.0;

        let mut ticks = Vec::new();
        let mut value = axis_min;
        while value <= axis_max + step * 0.5 {
            ticks.push(if value.abs() < step * 1e-6 {
                0.0
            } else {
                value
            });
            value += step;
        }

        return (axis_min, axis_max, step, ticks);
    }

    let range = nice_number(data_max - data_min, false);
    let step = nice_number(range / (target_ticks.saturating_sub(1).max(1) as f64), true);
    let axis_min = (data_min / step).floor() * step;
    let axis_max = (data_max / step).ceil() * step;

    let mut ticks = Vec::new();
    let mut value = axis_min;
    while value <= axis_max + step * 0.5 {
        let rounded = if value.abs() < step * 1e-6 {
            0.0
        } else {
            value
        };
        ticks.push(rounded);
        value += step;
    }

    (axis_min, axis_max, step, ticks)
}

fn nice_number(value: f64, round: bool) -> f64 {
    let exponent = value.abs().log10().floor();
    let fraction = value / 10f64.powf(exponent);

    let nice_fraction = if round {
        if fraction < 1.5 {
            1.0
        } else if fraction < 3.0 {
            2.0
        } else if fraction < 7.0 {
            5.0
        } else {
            10.0
        }
    } else if fraction <= 1.0 {
        1.0
    } else if fraction <= 2.0 {
        2.0
    } else if fraction <= 5.0 {
        5.0
    } else {
        10.0
    };

    nice_fraction * 10f64.powf(exponent)
}

fn format_axis_tick(value: f64, step: f64) -> String {
    let decimals = if step >= 1.0 {
        0usize
    } else {
        (-step.abs().log10()).ceil().max(0.0) as usize
    };

    let value = if value.abs() < step.abs() * 1e-6 {
        0.0
    } else {
        value
    };

    format!("{value:.decimals$}")
}

fn draw_thick_line_segment(
    img: &mut RgbaImage,
    start: (f32, f32),
    end: (f32, f32),
    thickness: i32,
    color: Rgba<u8>,
) {
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let length = (dx * dx + dy * dy).sqrt();
    if length <= f32::EPSILON {
        draw_antialiased_line_segment_mut(
            img,
            (start.0.round() as i32, start.1.round() as i32),
            (end.0.round() as i32, end.1.round() as i32),
            color,
            interpolate,
        );
        return;
    }

    let normal_x = -dy / length;
    let normal_y = dx / length;
    for offset in -thickness..=thickness {
        let offset = offset as f32;
        let offset_x = normal_x * offset;
        let offset_y = normal_y * offset;
        draw_antialiased_line_segment_mut(
            img,
            (
                (start.0 + offset_x).round() as i32,
                (start.1 + offset_y).round() as i32,
            ),
            (
                (end.0 + offset_x).round() as i32,
                (end.1 + offset_y).round() as i32,
            ),
            color,
            interpolate,
        );
    }
}

fn draw_dotted_horizontal_line(
    img: &mut RgbaImage,
    x0: f32,
    x1: f32,
    y: f32,
    thickness: i32,
    dash_len: f32,
    gap_len: f32,
    color: Rgba<u8>,
) {
    let mut x = x0;
    while x < x1 {
        let dash_end = (x + dash_len).min(x1);
        draw_thick_line_segment(img, (x, y), (dash_end, y), thickness, color);
        x = dash_end + gap_len;
    }
}

fn draw_vertical_text_centered(
    img: &mut RgbaImage,
    font: &FontArc,
    center_x: i32,
    center_y: i32,
    scale: f32,
    text: &str,
    color: Rgba<u8>,
) {
    let (width, height) = text_size(scale, font, text);
    if width == 0 || height == 0 {
        return;
    }

    let padding = 8u32;
    let mut text_img = RgbaImage::from_pixel(
        width + padding * 2,
        height + padding * 2,
        Rgba([0, 0, 0, 0]),
    );
    draw_text_mut(
        &mut text_img,
        color,
        padding as i32,
        padding as i32,
        scale,
        font,
        text,
    );

    let rotated = rotate270(&text_img);
    let start_x = center_x - rotated.width() as i32 / 2;
    let start_y = center_y - rotated.height() as i32 / 2;
    overlay_rgba(img, &rotated, start_x, start_y);
}

fn overlay_rgba(dst: &mut RgbaImage, src: &RgbaImage, x: i32, y: i32) {
    for (sx, sy, pixel) in src.enumerate_pixels() {
        let dx = x + sx as i32;
        let dy = y + sy as i32;
        if dx < 0 || dy < 0 || dx >= dst.width() as i32 || dy >= dst.height() as i32 {
            continue;
        }

        let alpha = pixel[3] as f32 / 255.0;
        if alpha <= 0.0 {
            continue;
        }

        let dst_px = dst.get_pixel_mut(dx as u32, dy as u32);
        for channel in 0..4 {
            let bg = dst_px[channel] as f32;
            let fg = pixel[channel] as f32;
            dst_px[channel] = ((fg * alpha) + (bg * (1.0 - alpha))).round() as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_plot_data_tracks_hits_and_curve() {
        let ranks = RankedList::new(
            vec!["g1".into(), "g2".into(), "g3".into(), "g4".into()],
            vec![2.0, 1.0, -1.0, -2.0],
        );
        let pathway = Pathway {
            name: "PW_A".into(),
            description: None,
            genes: vec!["g1".into(), "g2".into()],
        };

        let data = build_enrichment_plot_data(&ranks, &pathway, ScoreType::Std, 1.0).unwrap();
        assert_eq!(data.hit_positions, vec![1, 2]);
        assert_eq!(data.running_score.len(), 5);
        assert!(data.es > 0.0);
    }
}
