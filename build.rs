fn main() {
    // Vendor the protobuf compiler so users don't need system protoc (LanceDB/lance build deps need it).
    std::env::set_var("PROTOC", protobuf_src::protoc());
}
