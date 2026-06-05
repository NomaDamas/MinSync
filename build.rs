fn main() {
    #[cfg(not(windows))]
    {
        std::env::set_var("PROTOC", protobuf_src::protoc());
    }

    #[cfg(windows)]
    {
        println!("cargo:rerun-if-env-changed=PROTOC");
    }
}
