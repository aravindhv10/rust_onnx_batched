{pkgs ? import <nixpkgs> {}} :

let

mylist = with pkgs; [

bc
bison
blend2d
cargo
cargo-info
ffmpeg
ffmpeg.dev
fish
flex
fontconfig
fontconfig.dev
fontconfig.lib
gnumake
libelf
nasm
openssl
openssl.dev
pkg-config
python313Full
udev
zsh
zstd

] ;

in

(pkgs.mkShell {

name = "good_rust_env";

packages = mylist;

runScript = "fish";

})
