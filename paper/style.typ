// Paper layout for pandoc's typst writer: `pandoc ... -V template=style.typ`.
// Follows the NeurIPS style: US Letter, 5.5 x 9 in text block with a 1.5 in
// left margin, 10 pt Times-style type on 11 pt leading, paragraphs separated by
// half a line without indentation, a one-paragraph abstract indented 0.5 in on
// both sides under a centred 12 pt bold heading. STIX Two is an open
// Times-compatible face. Tables are booktabs-style; the running head carries a
// small project mark.

#let ink = rgb("#20252b")
#let muted = rgb("#626971")
#let accent = rgb("#20252b")
#let repo = "https://github.com/janmus1c/ncm-mlx"

#let conf(
  title: none,
  subtitle: none,
  authors: (),
  keywords: (),
  date: none,
  abstract-title: none,
  abstract: none,
  thanks: none,
  cols: 1,
  margin: (left: 1.5in, right: 1.5in, top: 1in, bottom: 1in),
  paper: "us-letter",
  lang: "en",
  region: "GB",
  font: ("STIX Two Text",),
  fontsize: 10pt,
  mathfont: ("STIX Two Math",),
  codefont: ("DejaVu Sans Mono",),
  linestretch: 1,
  sectionnumbering: "1.1",
  pagenumbering: "1",
  linkcolor: none,
  citecolor: none,
  filecolor: none,
  doc,
) = {
  set document(title: title, keywords: keywords)
  set text(lang: lang, region: region, font: font, size: fontsize, fill: ink)
  show math.equation: set text(font: mathfont)
  show raw: set text(font: codefont, size: 0.82em)
  set par(justify: true, leading: 0.42em, spacing: 5.5pt + 0.42em)

  let running-head = context {
    if counter(page).get().first() > 1 {
      set text(size: 8pt, fill: muted)
      grid(
        columns: (auto, 1fr, auto),
        column-gutter: 6pt,
        align: horizon,
        box(image("assets/logo.svg", height: 9pt)),
        smallcaps[ncm-mlx · working paper],
        emph[Next-concept modeling in MLX],
      )
      v(-4pt)
      line(length: 100%, stroke: 0.4pt + muted)
    }
  }
  set page(
    paper: paper,
    margin: margin,
    header: running-head,
    numbering: pagenumbering,
    footer: context align(center, text(size: 8.5pt, fill: muted, counter(page).display())),
  )

  set heading(numbering: sectionnumbering)
  show heading: set block(above: 1.5em, below: 0.8em)
  show heading.where(level: 1): it => {
    set text(size: 12pt, weight: "bold")
    it
  }
  show heading.where(level: 2): set text(size: 10pt, weight: "bold")
  show heading.where(level: 3): set text(size: 10pt, weight: "bold", style: "italic")

  show link: it => text(fill: accent, it)
  show figure.caption: it => {
    set text(size: 9pt)
    set par(justify: true)
    block(sticky: it.position == top)[#strong[#it.supplement #context it.counter.display(it.numbering):] #it.body]
  }
  set figure(gap: 0.8em)
  show figure: set block(above: 1.4em, below: 1.4em)
  set table(stroke: (x, y) => (
    top: if y == 0 { 0.8pt + ink } else { none },
    bottom: if y == 0 { 0.5pt + ink } else { none },
  ), inset: (x: 6pt, y: 4.5pt))
  set table.cell(breakable: false)
  show table: set text(size: 9pt)
  show table.cell: set align(left)
  show table.cell: set par(justify: false)
  show figure.where(kind: table): set block(breakable: true)
  show figure.where(kind: table): it => block(
    breakable: true, stroke: (bottom: 0.8pt + ink), inset: (bottom: 3pt), it,
  )
  show raw.where(block: true): it => block(
    fill: rgb("#f2f3f5"), inset: 8pt, radius: 3pt, width: 100%, it,
  )
  show quote: it => block(
    inset: (left: 10pt, y: 4pt), stroke: (left: 1.5pt + accent), text(size: 9.5pt, it.body),
  )

  // Title block.
  grid(
    columns: (auto, 1fr),
    column-gutter: 8pt,
    align: horizon,
    box(image("assets/logo.svg", height: 20pt)),
    text(size: 8.5pt, fill: muted)[
      #smallcaps[ncm-mlx · working paper]#if date != none [ · #date] \
      #link(repo)[#repo.slice(8)]
    ],
  )
  v(1.6em)
  align(center)[
    #line(length: 100%, stroke: 2pt + ink)
    #v(0.5em)
    #block(width: 100%, text(size: 17pt, weight: "bold", hyphenate: false, title))
    #v(0.3em)
    #line(length: 100%, stroke: 0.6pt + ink)
    #if subtitle != none { v(0.2em); text(size: 11.5pt, subtitle) }
    #v(1em)
    #for a in authors {
      block(spacing: 0.4em)[
        #text(size: 11pt, a.name) \
        #if a.affiliation != [] and a.affiliation != "" { text(size: 9pt, fill: muted, a.affiliation) }
      ]
    }
  ]
  if abstract != none {
    v(1.4em)
    pad(x: 0.5in, block(width: 100%)[
      #align(center, text(size: 12pt, weight: "bold")[#if abstract-title != none { abstract-title } else [Abstract]])
      #v(0.3em)
      #set text(size: 10pt)
      #set par(justify: true)
      #align(left, abstract)
    ])
  }
  if keywords.len() > 0 {
    v(0.6em)
    pad(x: 0.5in, text(size: 9pt)[*Keywords:* #keywords.join(", ")])
  }
  v(1.8em)
  doc
}
