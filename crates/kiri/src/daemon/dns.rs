//! `*.kiri.local`: a tiny DNS server on loopback that answers every name
//! under `kiri.local` with 127.0.0.1, where the domain router (domains.rs)
//! picks the container from the Host header. macOS sends these lookups here
//! because of `/etc/resolver/kiri.local`, which `kiri setup` installs once
//! (it needs an administrator). Anything else is refused.
//!
//! Also the relay for container lookups the VM's gateway resolver fails
//! (`serve_relay`): the guest agent sends them here over vsock, and they're
//! asked of the Mac's own resolvers.

use std::io::{self, Read, Write};
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// The resolver file's port; unprivileged, so the daemon needs no root.
pub const PORT: u16 = 53535;
pub const RESOLVER: &str = "/etc/resolver/kiri.local";
pub const SUFFIX: &str = "kiri.local";

pub fn start() -> io::Result<()> {
    let sock = UdpSocket::bind(("127.0.0.1", PORT))?;
    std::thread::Builder::new().name("kiri-dns".into()).spawn(move || {
        let mut buf = [0u8; 512];
        loop {
            let Ok((n, from)) = sock.recv_from(&mut buf) else { continue };
            if let Some(reply) = answer(&buf[..n]) {
                let _ = sock.send_to(&reply, from);
            }
        }
    })?;
    Ok(())
}

/// True when macOS will send `*.kiri.local` lookups to Kiri.
pub fn installed() -> bool {
    std::fs::read_to_string(RESOLVER).is_ok_and(|s| s.contains(&format!("port {PORT}")))
}

/// One reply for one query: A → 127.0.0.1 under kiri.local, an empty answer
/// for other types there (so AAAA falls back to A), REFUSED elsewhere.
fn answer(q: &[u8]) -> Option<Vec<u8>> {
    if q.len() < 12 || q[2] & 0x80 != 0 || u16::from_be_bytes([q[4], q[5]]) != 1 {
        return None;
    }
    // The question name, as labels.
    let mut i = 12;
    let mut name = String::new();
    loop {
        let len = *q.get(i)? as usize;
        i += 1;
        if len == 0 {
            break;
        }
        if len > 63 {
            return None;
        }
        let label = q.get(i..i + len)?;
        if !name.is_empty() {
            name.push('.');
        }
        name.push_str(&String::from_utf8_lossy(label).to_ascii_lowercase());
        i += len;
    }
    let qtype = u16::from_be_bytes([*q.get(i)?, *q.get(i + 1)?]);
    let question = q.get(12..i + 4)?;
    let ours = name == SUFFIX || name.ends_with(&format!(".{SUFFIX}"));
    let a = ours && qtype == 1;
    let mut r = Vec::with_capacity(question.len() + 28);
    r.extend_from_slice(&q[..2]);
    // QR, AA, RD copied; RCODE 5 (REFUSED) when the name is not ours.
    r.extend_from_slice(&[0x84 | (q[2] & 0x01), if ours { 0 } else { 5 }]);
    r.extend_from_slice(&[0, 1, 0, a as u8, 0, 0, 0, 0]);
    r.extend_from_slice(question);
    if a {
        // A pointer to the question name, class IN, TTL 60, 127.0.0.1.
        r.extend_from_slice(&[0xc0, 12, 0, 1, 0, 1, 0, 0, 0, 60, 0, 4, 127, 0, 0, 1]);
    }
    Some(r)
}

/// How long one resolver gets to answer a relayed lookup.
const RELAY_TRY: Duration = Duration::from_millis(1500);
/// All the resolvers together: the agent stops waiting at five seconds.
const RELAY_BUDGET: Duration = Duration::from_millis(3500);
/// How long the agent gets to send its query, and to take the reply.
const RELAY_IO: Duration = Duration::from_secs(5);
/// Relayed lookups at once, as many as the agent itself keeps in flight;
/// more are dropped (the client asks again).
const MAX_RELAYS: usize = 256;
static RELAYS: AtomicUsize = AtomicUsize::new(0);

/// One lookup from the guest agent: `u16 len | query` in, `u16 len | reply`
/// out (kiri_proto::DNS_RELAY_PORT). The agent sends here only what the
/// VM's gateway resolver refused or left unanswered, so this is the path
/// while macOS's resolver for the VM network is down.
pub fn serve_relay(stream: kiri_vmm::VsockStream) {
    if RELAYS.fetch_add(1, Ordering::Relaxed) >= MAX_RELAYS {
        RELAYS.fetch_sub(1, Ordering::Relaxed);
        return;
    }
    let spawned = std::thread::Builder::new().name("kiri-dns-relay".into()).spawn(move || {
        relay_one(&stream);
        RELAYS.fetch_sub(1, Ordering::Relaxed);
    });
    if spawned.is_err() {
        RELAYS.fetch_sub(1, Ordering::Relaxed);
    }
}

fn relay_one(stream: &kiri_vmm::VsockStream) {
    let Ok(fd) = stream.fd.try_clone() else { return };
    // A peer that connects and then says nothing doesn't keep a thread.
    // Best effort: without the timeouts the lookup still works.
    if let Err(e) = set_timeouts(fd.as_raw_fd(), RELAY_IO) {
        crate::diag::debug!("DNS relay: no socket timeouts", error = e);
    }
    let mut conn = std::fs::File::from(fd);
    let mut len = [0u8; 2];
    if conn.read_exact(&mut len).is_err() {
        return;
    }
    let mut query = vec![0u8; u16::from_be_bytes(len) as usize];
    if conn.read_exact(&mut query).is_err() || !is_query(&query) {
        return;
    }
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        crate::diag::warn!("the VM's DNS at its gateway did not answer; container lookups go through the Mac");
    }
    let name = question_name(&query).unwrap_or_default();
    let servers = mac_resolvers(&name, Path::new("/etc/resolv.conf"), Path::new("/etc/resolver"));
    let reply = relay_query(&query, &servers).unwrap_or_else(|| servfail(&query));
    let mut out = Vec::with_capacity(2 + reply.len());
    out.extend_from_slice(&(reply.len().min(u16::MAX as usize) as u16).to_be_bytes());
    out.extend_from_slice(&reply[..reply.len().min(u16::MAX as usize)]);
    let _ = conn.write_all(&out);
}

/// Whether `q` is a standard DNS query with one well-formed question: the
/// only thing the agent sends, and so all that's passed to the Mac's
/// resolvers (some of which listen on the Mac's loopback).
fn is_query(q: &[u8]) -> bool {
    q.len() >= 12
        && q[2] & 0x80 == 0 // QR: a query
        && (q[2] >> 3) & 0x0f == 0 // OPCODE: QUERY
        && u16::from_be_bytes([q[4], q[5]]) == 1
        && q[6..8] == [0, 0] // no answers
        && question_end(q).is_some()
}

fn set_timeouts(fd: std::os::fd::RawFd, t: Duration) -> io::Result<()> {
    let tv = libc::timeval { tv_sec: t.as_secs() as _, tv_usec: t.subsec_micros() as _ };
    let len = std::mem::size_of::<libc::timeval>() as libc::socklen_t;
    for opt in [libc::SO_RCVTIMEO, libc::SO_SNDTIMEO] {
        // SAFETY: a valid timeval for a socket option on a descriptor we hold.
        if unsafe { libc::setsockopt(fd, libc::SOL_SOCKET, opt, (&tv as *const libc::timeval).cast(), len) } != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// The first answer to `query` from `servers`, tried in order.
fn relay_query(query: &[u8], servers: &[SocketAddr]) -> Option<Vec<u8>> {
    let start = Instant::now();
    let mut buf = vec![0u8; 65_535];
    for server in servers {
        let left = RELAY_BUDGET.checked_sub(start.elapsed())?;
        let bind: SocketAddr = if server.is_ipv4() { ([0, 0, 0, 0], 0).into() } else { (std::net::Ipv6Addr::UNSPECIFIED, 0).into() };
        let Ok(sock) = UdpSocket::bind(bind) else { continue };
        if sock.connect(server).is_err() || sock.send(query).is_err() {
            continue;
        }
        let until = Instant::now() + RELAY_TRY.min(left);
        // Read past stray datagrams until this query's answer or the deadline.
        while let Some(wait) = until.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) {
            if sock.set_read_timeout(Some(wait)).is_err() {
                break;
            }
            let Ok(n) = sock.recv(&mut buf) else { break };
            let r = &buf[..n];
            if n >= 12 && r[..2] == query[..2] && r[2] & 0x80 != 0 {
                return Some(r.to_vec());
            }
        }
    }
    None
}

/// SERVFAIL for `query`: no resolver on the Mac answered either.
fn servfail(query: &[u8]) -> Vec<u8> {
    let end = question_end(query).unwrap_or(12).min(query.len());
    let mut r = query[..end].to_vec();
    r[2] = 0x80 | (query[2] & 0x79); // QR, opcode and RD kept
    r[3] = 0x80 | 2; // RA, SERVFAIL
    r[6..12].fill(0);
    if end > 12 {
        r[5] = 1;
    } else {
        r[4..6].fill(0);
    }
    r
}

/// Offset just past the first question, if there is one.
fn question_end(q: &[u8]) -> Option<usize> {
    if q.len() < 12 || u16::from_be_bytes([q[4], q[5]]) == 0 {
        return None;
    }
    let mut i = 12;
    loop {
        let len = *q.get(i)? as usize;
        if len == 0 {
            break;
        }
        if len > 63 {
            return None;
        }
        i += 1 + len;
    }
    (i + 5 <= q.len()).then_some(i + 5)
}

/// The first question's name, lowercase, without the trailing dot.
fn question_name(q: &[u8]) -> Option<String> {
    question_end(q)?;
    let mut i = 12;
    let mut name = String::new();
    loop {
        let len = q[i] as usize;
        if len == 0 {
            return Some(name);
        }
        if !name.is_empty() {
            name.push('.');
        }
        name.push_str(&String::from_utf8_lossy(&q[i + 1..i + 1 + len]).to_ascii_lowercase());
        i += 1 + len;
    }
}

/// Where the Mac would send a lookup for `name`: the servers of the
/// `/etc/resolver/<domain>` file with the longest domain that `name` is in
/// (VPNs and tools such as dnsmasq set these up), else `/etc/resolv.conf`'s,
/// which macOS keeps to its current primary DNS servers. Read on every
/// lookup, so network changes are followed. Kiri's own `kiri.local` file is
/// skipped: its server answers 127.0.0.1, the Mac's loopback, which means
/// nothing in a container.
fn mac_resolvers(name: &str, resolv_conf: &Path, resolver_dir: &Path) -> Vec<SocketAddr> {
    let mut best: Option<(usize, Vec<SocketAddr>)> = None;
    for entry in std::fs::read_dir(resolver_dir).into_iter().flatten().flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else { continue };
        let file = entry.file_name().to_string_lossy().to_ascii_lowercase();
        let domain = text
            .lines()
            .find_map(|l| l.trim().strip_prefix("domain").map(|d| d.trim().trim_end_matches('.').to_ascii_lowercase()))
            .filter(|d| !d.is_empty())
            .unwrap_or(file);
        if domain == SUFFIX {
            continue;
        }
        let inside = name == domain || name.ends_with(&format!(".{domain}"));
        if !inside || best.as_ref().is_some_and(|(len, _)| *len >= domain.len()) {
            continue;
        }
        let servers = parse_servers(&text);
        if !servers.is_empty() {
            best = Some((domain.len(), servers));
        }
    }
    best.map(|(_, s)| s).unwrap_or_else(|| parse_servers(&std::fs::read_to_string(resolv_conf).unwrap_or_default()))
}

/// `nameserver` lines (and a `port` line) of a resolv.conf-style file.
/// Link-local IPv6 servers, which need an interface named, are skipped.
fn parse_servers(text: &str) -> Vec<SocketAddr> {
    let field = |key: &'static str| text.lines().filter_map(move |l| l.trim().strip_prefix(key).filter(|v| v.starts_with([' ', '\t'])).map(str::trim));
    let port = field("port").find_map(|p| p.parse::<u16>().ok()).unwrap_or(53);
    field("nameserver").filter_map(|ip| ip.parse::<IpAddr>().ok()).map(|ip| SocketAddr::new(ip, port)).collect()
}

#[cfg(test)]
mod tests {
    use super::answer;
    use std::net::UdpSocket;

    fn query(name: &str, qtype: u16) -> Vec<u8> {
        let mut q = vec![0x12, 0x34, 0x01, 0x00, 0, 1, 0, 0, 0, 0, 0, 0];
        for l in name.split('.') {
            q.push(l.len() as u8);
            q.extend_from_slice(l.as_bytes());
        }
        q.push(0);
        q.extend_from_slice(&qtype.to_be_bytes());
        q.extend_from_slice(&[0, 1]);
        q
    }

    #[test]
    fn answers_kiri_local_only() {
        let r = answer(&query("Web.kiri.local", 1)).unwrap();
        assert_eq!((&r[..2], r[3], r[7]), (&[0x12, 0x34][..], 0, 1));
        assert_eq!(&r[r.len() - 4..], &[127, 0, 0, 1]);
        let aaaa = answer(&query("web.kiri.local", 28)).unwrap();
        assert_eq!((aaaa[3], aaaa[7]), (0, 0));
        assert_eq!(answer(&query("example.com", 1)).unwrap()[3], 5);
        assert_eq!(answer(&query("notkiri.local", 1)).unwrap()[3], 5);
        assert!(answer(&[0; 5]).is_none());
    }

    #[test]
    fn relayed_lookups_go_where_the_mac_would_send_them() {
        let dir = std::env::temp_dir().join(format!("kiri-dns-test-{}", std::process::id()));
        let resolvers = dir.join("resolver");
        std::fs::create_dir_all(&resolvers).unwrap();
        let conf = dir.join("resolv.conf");
        std::fs::write(&conf, "# macOS\nsearch home\nnameserver 192.168.1.1\nnameserver fe80::1%en0\nnameserver 2001:db8::53\n").unwrap();
        std::fs::write(resolvers.join("corp.example"), "nameserver 10.0.0.53\n").unwrap();
        std::fs::write(resolvers.join("eu.corp.example"), "nameserver 10.1.0.53\nport 5353\n").unwrap();
        std::fs::write(resolvers.join("kiri.local"), "nameserver 127.0.0.1\nport 53535\n").unwrap();
        let r = |name: &str| super::mac_resolvers(name, &conf, &resolvers).iter().map(ToString::to_string).collect::<Vec<_>>();
        assert_eq!(r("registry-1.docker.io"), ["192.168.1.1:53", "[2001:db8::53]:53"]);
        assert_eq!(r("git.corp.example"), ["10.0.0.53:53"]);
        assert_eq!(r("corp.example"), ["10.0.0.53:53"]);
        assert_eq!(r("db.eu.corp.example"), ["10.1.0.53:5353"], "the longest domain wins");
        assert_eq!(r("notcorp.example"), ["192.168.1.1:53", "[2001:db8::53]:53"]);
        assert_eq!(r("web.kiri.local"), ["192.168.1.1:53", "[2001:db8::53]:53"], "Kiri's own resolver is the Mac's loopback");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_relayed_lookup_takes_the_first_resolver_that_answers() {
        let up = UdpSocket::bind("127.0.0.1:0").unwrap();
        let addr = up.local_addr().unwrap();
        let resolver = std::thread::spawn(move || {
            let mut buf = [0u8; 512];
            let (n, from) = up.recv_from(&mut buf).unwrap();
            // A stray datagram first, then the answer.
            up.send_to(&[9; 12], from).unwrap();
            buf[2] |= 0x80;
            up.send_to(&buf[..n], from).unwrap();
        });
        // A closed port first: refused, then the next one is asked.
        let closed = UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let q = query("registry-1.docker.io", 1);
        let r = super::relay_query(&q, &[closed, addr]).expect("answered");
        resolver.join().unwrap();
        assert_eq!((&r[..2], r[2] & 0x80, &r[12..]), (&q[..2], 0x80, &q[12..]));
        assert_eq!(super::question_name(&q).as_deref(), Some("registry-1.docker.io"));
    }

    #[test]
    fn only_standard_queries_are_relayed() {
        let q = query("registry-1.docker.io", 1);
        assert!(super::is_query(&q));
        let mut reply = q.clone();
        reply[2] |= 0x80;
        assert!(!super::is_query(&reply), "a response");
        let mut update = q.clone();
        update[2] = 5 << 3;
        assert!(!super::is_query(&update), "an UPDATE");
        let mut two = q.clone();
        two[5] = 2;
        assert!(!super::is_query(&two), "two questions");
        assert!(!super::is_query(&q[..q.len() - 1]), "cut short");
        assert!(!super::is_query(&[0x12, 0x34, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0xc0, 12, 0, 1, 0, 1]), "a compression pointer");
        assert!(!super::is_query(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"), "not DNS at all");
    }

    #[test]
    fn servfail_keeps_the_id_and_question() {
        let q = query("example.com", 1);
        let r = super::servfail(&q);
        assert_eq!((&r[..2], r[2] & 0x80, r[3] & 0x0f), (&q[..2], 0x80, 2));
        assert_eq!((&r[4..12], &r[12..]), (&[0, 1, 0, 0, 0, 0, 0, 0][..], &q[12..]));
        assert_eq!(super::servfail(&[0x12, 0x34, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])[3], 0x82);
    }
}
