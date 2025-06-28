use kornia::image::Image;
use kornia::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8("./image.jpg")?;

    println!("Hello, world! 🦀");
    println!("Loaded Image size: {:?}", image.size());
    println!("\nGoodbyte!");

    Ok(())
}

