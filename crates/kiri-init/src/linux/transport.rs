//! How the agent reaches the host. In the VM that is vsock. On a Linux host
//! the agent runs as an ordinary root process beside the daemon (no VM), and
//! the same ports are unix sockets in `KIRI_AGENT_DIR`: the agent listens
//! on `g<port>.sock` and dials the daemon's `h<port>.sock`.

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::PathBuf;
use std::sync::OnceLock;

use super::sys;

const VMADDR_CID_HOST: u32 = 2;

/// The unix-socket directory when running as a host process.
pub fn host_dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("KIRI_AGENT_DIR").map(PathBuf::from)).as_ref()
}

/// True when the agent runs on a Linux host rather than as the VM's PID 1.
pub fn is_host_process() -> bool {
    host_dir().is_some()
}

fn unix_addr(path: &std::path::Path) -> io::Result<(libc::sockaddr_un, libc::socklen_t)> {
    // SAFETY: sockaddr_un is plain data; all-zero is valid.
    let mut a: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    a.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = std::os::unix::ffi::OsStrExt::as_bytes(path.as_os_str());
    if bytes.len() >= a.sun_path.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("socket path too long: {}", path.display())));
    }
    for (d, s) in a.sun_path.iter_mut().zip(bytes) {
        *d = *s as libc::c_char;
    }
    Ok((a, std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t))
}

/// A non-blocking listener for `port`.
pub fn listen(port: u32) -> io::Result<OwnedFd> {
    // SAFETY: plain socket setup; the fd is owned immediately.
    unsafe {
        if let Some(dir) = host_dir() {
            let path = dir.join(format!("g{port}.sock"));
            let _ = std::fs::remove_file(&path);
            let fd = sys::check(libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK, 0) as _, "unix socket")?;
            let sock = OwnedFd::from_raw_fd(fd as RawFd);
            let (a, len) = unix_addr(&path)?;
            sys::check(libc::bind(sock.as_raw_fd(), (&a as *const libc::sockaddr_un).cast(), len) as _, "bind")?;
            sys::check(libc::listen(sock.as_raw_fd(), 128) as _, "listen")?;
            return Ok(sock);
        }
        let fd = sys::check(libc::socket(libc::AF_VSOCK, libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK, 0) as _, "vsock socket")?;
        let sock = OwnedFd::from_raw_fd(fd as RawFd);
        let mut addr: libc::sockaddr_vm = std::mem::zeroed();
        addr.svm_family = libc::AF_VSOCK as libc::sa_family_t;
        addr.svm_cid = libc::VMADDR_CID_ANY;
        addr.svm_port = port;
        sys::check(libc::bind(sock.as_raw_fd(), (&addr as *const libc::sockaddr_vm).cast(), std::mem::size_of::<libc::sockaddr_vm>() as u32) as _, "vsock bind")?;
        sys::check(libc::listen(sock.as_raw_fd(), 128) as _, "vsock listen")?;
        Ok(sock)
    }
}

/// A blocking connection to the host's `port`.
pub fn dial(port: u32) -> io::Result<OwnedFd> {
    // SAFETY: plain socket setup and connect; the fd is owned immediately.
    unsafe {
        if let Some(dir) = host_dir() {
            let fd = sys::check(libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) as _, "unix socket")?;
            let sock = OwnedFd::from_raw_fd(fd as RawFd);
            let (a, len) = unix_addr(&dir.join(format!("h{port}.sock")))?;
            sys::check(libc::connect(sock.as_raw_fd(), (&a as *const libc::sockaddr_un).cast(), len) as _, "connect")?;
            return Ok(sock);
        }
        let fd = sys::check(libc::socket(libc::AF_VSOCK, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) as _, "vsock socket")?;
        let sock = OwnedFd::from_raw_fd(fd as RawFd);
        let mut addr: libc::sockaddr_vm = std::mem::zeroed();
        addr.svm_family = libc::AF_VSOCK as libc::sa_family_t;
        addr.svm_cid = VMADDR_CID_HOST;
        addr.svm_port = port;
        sys::check(libc::connect(sock.as_raw_fd(), (&addr as *const libc::sockaddr_vm).cast(), std::mem::size_of::<libc::sockaddr_vm>() as u32) as _, "vsock connect")?;
        Ok(sock)
    }
}

/// `SO_VM_SOCKETS_CONNECT_TIMEOUT` (linux/vm_sockets.h, the `_OLD` value,
/// which takes a native `timeval`; libc doesn't name it).
const SO_VM_SOCKETS_CONNECT_TIMEOUT: libc::c_int = 6;

/// Like `dial`, but a connection the host doesn't take within `timeout`
/// fails (vsock waits two seconds by default), for callers in the agent's
/// loop, which nothing else runs in while it waits.
pub fn dial_within(port: u32, timeout: std::time::Duration) -> io::Result<OwnedFd> {
    if host_dir().is_some() {
        return dial(port); // a unix socket: connect never waits
    }
    // SAFETY: plain socket setup and connect; the fd is owned immediately.
    unsafe {
        let fd = sys::check(libc::socket(libc::AF_VSOCK, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) as _, "vsock socket")?;
        let sock = OwnedFd::from_raw_fd(fd as RawFd);
        let tv = libc::timeval { tv_sec: timeout.as_secs() as _, tv_usec: timeout.subsec_micros() as _ };
        sys::check(
            libc::setsockopt(sock.as_raw_fd(), libc::AF_VSOCK, SO_VM_SOCKETS_CONNECT_TIMEOUT, (&tv as *const libc::timeval).cast(), std::mem::size_of::<libc::timeval>() as u32) as _,
            "vsock connect timeout",
        )?;
        let mut addr: libc::sockaddr_vm = std::mem::zeroed();
        addr.svm_family = libc::AF_VSOCK as libc::sa_family_t;
        addr.svm_cid = VMADDR_CID_HOST;
        addr.svm_port = port;
        sys::check(libc::connect(sock.as_raw_fd(), (&addr as *const libc::sockaddr_vm).cast(), std::mem::size_of::<libc::sockaddr_vm>() as u32) as _, "vsock connect")?;
        Ok(sock)
    }
}

/// The host's own DNS servers, for a host process (no DHCP lease to ask).
/// systemd-resolved's stub (127.0.0.53) is skipped for its upstreams.
pub fn host_dns() -> Vec<[u8; 4]> {
    let parse = |path: &str| -> Vec<[u8; 4]> {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| l.strip_prefix("nameserver"))
            .filter_map(|ip| ip.trim().parse::<std::net::Ipv4Addr>().ok())
            .filter(|ip| !ip.is_loopback())
            .map(|ip| ip.octets())
            .collect()
    };
    let direct = parse("/etc/resolv.conf");
    if direct.is_empty() { parse("/run/systemd/resolve/resolv.conf") } else { direct }
}
