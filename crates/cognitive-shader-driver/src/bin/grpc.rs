//! shader-grpc — debug gRPC server for live shader testing.
//!
//! ```bash
//! cargo run -p cognitive-shader-driver --features grpc --bin shader-grpc
//! ```

use std::sync::Arc;
use cognitive_shader_driver::bindspace::BindSpace;
use cognitive_shader_driver::driver::CognitiveShaderBuilder;
use cognitive_shader_driver::grpc::ShaderGrpcService;

use bgz17::base17::Base17;
use bgz17::palette::Palette;
use bgz17::palette_semiring::PaletteSemiring;

fn demo_palette() -> PaletteSemiring {
    let entries: Vec<Base17> = (0..256).map(|i| {
        let mut dims = [0i16; 17];
        dims[0] = (i * 100 % 3400) as i16;
        dims[1] = ((i * 37) % 200) as i16;
        dims[2] = ((i * 53) % 300) as i16;
        Base17 { dims }
    }).collect();
    PaletteSemiring::build(&Palette { entries })
}

#[tokio::main]
async fn main() {
    let bs = Arc::new(BindSpace::zeros(4096));
    let sr = Arc::new(demo_palette());
    let driver = CognitiveShaderBuilder::new()
        .bindspace(bs)
        .semiring(sr)
        .build();

    let addr = "0.0.0.0:50051".parse().unwrap();
    eprintln!("shader-grpc listening on {addr}");
    tonic::transport::Server::builder()
        .add_service(ShaderGrpcService::new(driver).into_server())
        .serve(addr)
        .await
        .unwrap();
}
