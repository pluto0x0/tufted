#import "../config.typ": template, tufted
#import "@preview/cmarker:0.1.7"
#show: template

= Tufted

#{
  let md-content = read("../assets/README.md")
  let md-content = md-content.trim(regex("\s*#.+?\n")) // Remove first-level heading

  // Render markdown content with custom image handling
  cmarker.render(
    md-content,
    scope: (
      image: (source, alt: none, format: auto) => figure(image(
        "../assets/" + source, // Modify paths for images
        alt: alt,
        format: format,
      )),
    ),
  )
}
