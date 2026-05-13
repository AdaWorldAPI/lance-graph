//! OCR contract. Zero-dep.

use core::future::Future;

pub trait OcrProvider: Send + Sync {
    type Doc;
    type Error: core::fmt::Debug + Send + Sync + 'static;

    fn recognize<'a>(
        &'a self,
        image: PageImage<'a>,
        opts: OcrOpts<'a>,
    ) -> impl Future<Output = Result<Self::Doc, Self::Error>> + Send + 'a;
}

pub struct PageImage<'a> {
    pub bytes: &'a [u8],
    pub mime: &'a str,
    pub page_index: u32,
    pub dpi_hint: Option<u16>,
}

pub struct OcrOpts<'a> {
    /// Expected languages, BCP-47. OCR engine may or may not honor.
    pub languages: &'a [&'a str],
    /// If true, the implementation should emit full layout blocks
    /// (paragraphs, tables) rather than just text.
    pub layout: bool,
    /// Confidence threshold below which tokens are dropped.
    pub min_confidence: f32,
}

/// Bounding box in image pixel space.
#[derive(Clone, Copy, Debug)]
pub struct Bbox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Semantic classification of a layout block.
#[derive(Clone, Copy, Debug)]
pub enum BlockKind {
    Text,
    Heading,
    Table,
    Figure,
    Signature,
    Stamp,
    Other,
}

pub struct LayoutBlock<'a> {
    pub kind: BlockKind,
    pub bbox: Bbox,
    pub text: &'a str,
    pub confidence: f32,
}
