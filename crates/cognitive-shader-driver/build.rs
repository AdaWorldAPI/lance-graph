fn main() {
    #[cfg(feature = "grpc")]
    {
        tonic_build::compile_protos("proto/shader.proto")
            .expect("Failed to compile shader.proto");
    }
}
