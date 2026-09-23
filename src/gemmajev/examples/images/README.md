# Image-style demo assets

These are illustrative Playground examples, not a benchmark or a test of whether an image was AI-generated.

All four samples are **AI-generated style examples**, created for GemmaJev on 2026-09-24 using Codex's built-in `image_gen` tool. Samples 1–3 each used one independent generation call. Sample 4 used one generation call followed by one image-editing call to make its background opaque white; the edit used only its own generated draft. They depict the same subject: an orange tabby cat sitting beside a blue ceramic mug by a window. No external imagery, reference photographs, user photos, stock imagery or named artist styles were supplied. Sample 1 has a photographic appearance; it is **not a real camera photograph**.

The generation prompts requested 1024 × 1024 images. The final tool outputs are 1254 × 1254 PNGs, copied into the repository unchanged, with no subsequent resizing, cropping, compositing or editing. Generation is nondeterministic; the checked-in files and hashes identify the exact examples used by the demo.

Sample 4 presents the cat and mug as an infographic with coordinate axes, measurement arrows and English annotations. These labels describe the subjects and schematic view; they do not name the image-style categories. Serving the bundled assets adds no third-party dependency to GemmaJev.

| File | Intended appearance | Dimensions | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| [sample_1.png](sample_1.png) | Photographic appearance | 1254 × 1254 | 2,016,166 | `48dc4f50f3427605a3f6373e1688ba1c4bbea1dda090fa4a13700be873fad8d4` |
| [sample_2.png](sample_2.png) | Watercolor painting | 1254 × 1254 | 2,817,718 | `fd1a777764457bcb29d357e77d7ab7a58234f3420120771c6c313a3c68c1bb7d` |
| [sample_3.png](sample_3.png) | Japanese 2D anime | 1254 × 1254 | 1,551,325 | `856ac05b190c5a7f93b6e81048e915dd9708357ef92292680ec215f61b5a7a2d` |
| [sample_4.png](sample_4.png) | Other: cat-and-mug infographic | 1254 × 1254 | 1,342,769 | `91bbbea4aafa5fc70cb9f27a1c869a073f1fab8a0c515a71c454d7e094304ecb` |

All four saved images were visually inspected for composition, intended appearance and absence of logos and watermarks. Samples 1–3 contain no text; sample 4 intentionally includes its title, callouts and axis and dimension labels.

## Generation and editing prompts

### sample_1.png

```text
Use case: photorealistic-natural. Asset type: an AI-generated visual-style example for an image-classification demo. Create one square image, 1024 by 1024 pixels. Subject: one orange tabby cat sitting on a windowsill, beside one blue ceramic mug. Simple, uncluttered indoor composition showing the whole seated cat and the mug, with a window visible. Style: convincing natural photography appearance, realistic individual fur and whiskers, natural proportions, soft daylight through the window, subtle optical depth of field and realistic ceramic texture. Keep the scene ordinary and the background quiet. No text, captions, letters, logos, brands, signature, border, or watermark. This is an AI-generated image in a photographic style, not a real camera photograph.
```

### sample_2.png

```text
Use case: illustration-story. Asset type: an AI-generated visual-style example for an image-classification demo. Create one square image, 1024 by 1024 pixels. Subject: one orange tabby cat sitting on a windowsill, beside one blue ceramic mug. Simple, uncluttered indoor composition showing the whole seated cat and the mug, with a window visible. Style/medium: unmistakable traditional watercolor painting on textured white watercolor paper. Use visible brush strokes, translucent layered pigment washes, soft pigment blooms, granulation, uneven painted edges and areas of untouched paper. Soft daylight is suggested through the painted window. The cat should read as a hand-painted animal with natural proportions, not an anime character or cartoon. No photographic fur rendering. No text, captions, letters, logos, brands, signature, border, or watermark.
```

### sample_3.png

```text
Use case: illustration-story. Asset type: an AI-generated visual-style example for an image-classification demo. Create one square image, 1024 by 1024 pixels. Subject: one orange tabby cat sitting on a windowsill, beside one blue ceramic mug. Simple, uncluttered indoor composition showing the whole seated cat and the mug, with a window visible. Style/medium: unmistakable Japanese 2D anime illustration. Make the cat a stylized anime animal character with expressive slightly enlarged eyes, clean deliberate dark outlines, simplified graphic fur tufts and stripes, flat color areas and distinct cel-shaded shadow shapes. Use a simple anime interior background and soft daylight from the window. Keep everything visibly drawn in 2D, with no photographic textures, realistic individual fur or watercolor paper texture. No text, captions, letters, logos, brands, signature, border, or watermark.
```

### sample_4.png

Generation:

```text
Use case: infographic-diagram.
Asset type: one raster sample for the GemmaJev image-style Playground, replacing its generic bar chart. Create one square 1024 by 1024 pixel image.
Primary request: depict the same subject as the other three demo samples—one seated orange tabby cat beside one blue ceramic mug on a windowsill—but render the entire scene as a clean technical infographic / schematic.
Composition: white background with a very faint coordinate grid. One large central front-view schematic shows the full seated cat, the mug, a simple windowsill baseline, and the outline of the window. Keep both subjects easily recognizable, with ample margins. Represent the cat with flat geometric orange shapes, a few dark stripe marks, simple eyes and whiskers; represent the mug as a flat blue cylinder with a handle. Use crisp uniform navy outlines, dashed alignment guides, ruler ticks, dimension arrows and two clear leader-line callouts. Diagram conventions should be prominent throughout the composition, not merely a border around an illustration. No realistic light, shadows, fur, painting texture, gradients, anime styling or photographic background.
Text (verbatim): a modest top title "CAT + MUG", a small subtitle "FRONT VIEW", callouts "CAT" and "MUG", short axis labels "x" and "y", and dimension labels "h1" and "h2". All text should be sharp English sans serif with generous spacing. Do not print any image-style category names such as photography, artwork, anime, other, chart, diagram, or infographic.
Constraints: exactly one cat and one mug, no people or additional scene objects, no logos, signatures or watermark. This must look like a polished explanatory graphic with geometric construction and annotations, rather than a cartoon scene. Output the finished image only.
```

Edit of the generated draft to produce the saved image:

```text
Edit the provided cat-and-mug schematic. Change only the background/transparency: composite the entire existing graphic over a solid pure white (#FFFFFF) background and make the finished image fully opaque. All pixels must have alpha 255; do not return a transparent cutout or a black background. Preserve the cat, mug, window outline, title CAT + MUG, FRONT VIEW subtitle, CAT and MUG callouts, coordinate axes, dimension lines, grid and all their positions and colors exactly. Keep the square composition and existing margins. Do not add, remove, redraw, crop or restyle any foreground content. Return the finished white-background diagram.
```
