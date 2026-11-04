//! Starts containers and execs.
//!
//! A container is one clone3 straight into fresh namespaces and its cgroup;
//! the child sets up its root and execs. An exec joins a running container's
//! namespaces and cgroup the same way. No runc, no shim, no extra fork:
//! every process is a direct child of the agent, so waiting just works.

use std::ffi::{CStr, CString};
use std::fs;
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use kiri_proto::{ExecSpec, RunSpec, StrList, net_opts, run_flags};

use super::inroot::{self, Last};
use super::layers;
use super::net::{self, Bridges};
use super::sys::{self, CLONE_INTO_CGROUP, CloneArgs, Received, child_fail};

/// Capabilities a container keeps; Docker's default set. All others are
/// dropped from the bounding set before exec.
const KEEP_CAPS: [u32; 14] = [
    0,  // CHOWN
    1,  // DAC_OVERRIDE
    3,  // FOWNER
    4,  // FSETID
    5,  // KILL
    6,  // SETGID
    7,  // SETUID
    8,  // SETPCAP
    10, // NET_BIND_SERVICE
    13, // NET_RAW
    18, // SYS_CHROOT
    27, // MKNOD
    29, // AUDIT_WRITE
    31, // SETFCAP
];
const LAST_CAP: u32 = 40;

const DEFAULT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

/// What networking the agent can offer a new container right now.
pub struct NetCtx<'a> {
    pub bridges: &'a mut Bridges,
    pub dns: &'a mut super::dns::Dns,
    /// The VM's upstream resolvers, once DHCP has delivered them.
    pub upstream_dns: &'a [[u8; 4]],
    /// For `--network container:<id>`: that container's PID 1, if running.
    pub join_pid: Option<libc::pid_t>,
}

/// Where containers live under the data directory.
#[derive(Debug)]
pub struct Paths {
    pub layers: PathBuf,
    pub containers: PathBuf,
    pub cgroup: PathBuf,
}

/// A process the agent started: a container's PID 1, or an exec.
#[derive(Debug)]
pub struct Process {
    /// Container id, or exec id for an exec.
    pub id: String,
    /// For an exec, the container it runs in.
    pub container: Option<String>,
    pub pid: libc::pid_t,
    /// Pipe read end, or the pty master for a TTY.
    pub stdout: Option<OwnedFd>,
    /// None for a TTY: the terminal merges both streams.
    pub stderr: Option<OwnedFd>,
    /// Write end for `Input`: a pipe, or a duplicate of the pty master.
    pub stdin: Option<OwnedFd>,
    pub tty: bool,
    /// (bridge, ip) registered with the DNS server, to drop on exit.
    pub dns_entries: Vec<(String, [u8; 4])>,
    /// Host sides of its veths. Deleted on exit: a container sharing this
    /// network namespace would otherwise keep them, and the address that
    /// goes back to the pool, alive.
    pub veths: Vec<String>,
    /// Inode of its own network namespace, or of the one it joined; 0 for
    /// the host's and for execs. Says whether a namespace outlives this
    /// container (see `teardown`).
    pub netns: u64,
    /// What to undo when a container exits; None for execs.
    teardown: Option<Teardown>,
}

#[derive(Debug)]
struct Teardown {
    base: PathBuf,
    rootfs: PathBuf,
    cgroup: PathBuf,
    remove: bool,
}

/// Everything the child needs, prepared before clone so the child does as
/// little as possible between clone and exec.
struct Prepared {
    argv: Vec<CString>,
    envp: Vec<CString>,
    path_dirs: Vec<Vec<u8>>,
    hostname: Option<Vec<u8>>,
    /// `--domainname`, set in the container's UTS namespace.
    domainname: Option<Vec<u8>>,
    /// `--network container:<id>`: that container's network namespace, joined
    /// instead of a new one.
    net_join: Option<fs::File>,
    /// Generated /etc files to bind over the image's: (source, target relative to the root).
    etc_files: Vec<(CString, CString)>,
    mounts: Vec<Mount>,
    /// For a bridged container: the child waits here until the agent has
    /// set up its network, so the program never starts without it.
    net_ready: Option<OwnedFd>,
    cwd: CString,
    user: String,
    net_none: bool,
    tty: bool,
    stdin: bool,
    privileged: bool,
    /// Capabilities kept, one bit each.
    caps: u64,
    seccomp: bool,
    no_new_privs: bool,
    readonly_rootfs: bool,
    shm_size: u64,
    /// `--init`: PID 1 supervises the program instead of being it.
    init: bool,
    /// Ulimits, groups and device nodes from the `opts` lines.
    host: HostApplied,
}

/// The parts of [`super::hostopts::HostOpts`] the child applies itself.
#[derive(Default)]
struct HostApplied {
    rlimits: Vec<(u32, u64, u64)>,
    groups: Vec<String>,
    nodes: Vec<super::hostopts::Node>,
}

/// Docker's default capability set as a mask.
fn default_caps() -> u64 {
    KEEP_CAPS.iter().fold(0, |m, c| m | 1 << c)
}

/// The child's ends of its standard streams.
enum ChildStdio {
    /// The child makes a pty inside its own /dev and sends the master back.
    Tty { sock: OwnedFd },
    Pipes { stdin: Option<OwnedFd>, stdout: OwnedFd, stderr: OwnedFd },
}

/// The agent's ends.
enum ParentStdio {
    Tty { sock: OwnedFd },
    Pipes { stdin: Option<OwnedFd>, stdout: OwnedFd, stderr: OwnedFd },
}

fn stdio(p: &Prepared) -> io::Result<(ParentStdio, ChildStdio)> {
    if p.tty {
        let (parent, child) = sys::socketpair()?;
        return Ok((ParentStdio::Tty { sock: parent }, ChildStdio::Tty { sock: child }));
    }
    let (out_r, out_w) = sys::pipe()?;
    let (err_r, err_w) = sys::pipe()?;
    let (in_w, in_r) = if p.stdin {
        let (r, w) = sys::pipe()?;
        sys::set_nonblocking(w.as_raw_fd())?;
        (Some(w), Some(r))
    } else {
        (None, None)
    };
    Ok((
        ParentStdio::Pipes { stdin: in_w, stdout: out_r, stderr: err_r },
        ChildStdio::Pipes { stdin: in_r, stdout: out_w, stderr: err_w },
    ))
}

/// After clone: collect the agent's ends. For a TTY this waits for the child
/// to send the pty master, or the reason it could not.
fn finish_stdio(parent: ParentStdio, child: ChildStdio) -> io::Result<(Option<OwnedFd>, Option<OwnedFd>, Option<OwnedFd>)> {
    drop(child);
    match parent {
        ParentStdio::Pipes { stdin, stdout, stderr } => Ok((stdin, Some(stdout), Some(stderr))),
        ParentStdio::Tty { sock } => match sys::recv_fd(sock.as_raw_fd())? {
            Received::Fd(master) => {
                sys::set_nonblocking(master.as_raw_fd())?;
                let input = master.try_clone()?;
                Ok((Some(input), Some(master), None))
            }
            Received::Message(m) if m.is_empty() => Err(io::Error::other("container exited before its terminal was ready")),
            Received::Message(m) => Err(io::Error::other(m)),
        },
    }
}

pub fn spawn(spec: &RunSpec<'_>, paths: &Paths, net: NetCtx<'_>) -> io::Result<Process> {
    if spec.argv.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "no command to run"));
    }
    let rootfs = mount_rootfs(paths, spec.id, spec.layers)?;
    let base = paths.containers.join(spec.id);

    let cgroup = paths.cgroup.join(spec.id);
    let remove = spec.flags & run_flags::REMOVE != 0;
    let undo = Teardown { base: base.clone(), rootfs: rootfs.clone(), cgroup: cgroup.clone(), remove };
    // A restart that fails must not take the stopped container's files with it.
    let cleanup = |e: io::Error| {
        undo_now(&undo, remove);
        e
    };
    // A previous run may have left sub-cgroups (Kubernetes, Docker in
    // Docker make them); its cgroup would then refuse processes.
    remove_cgroup(&cgroup);
    if !paths.cgroup.is_dir() {
        super::containers_cgroup(&paths.cgroup);
    }
    let _ = fs::create_dir(&cgroup);
    if spec.limits != kiri_proto::Limits::default() {
        super::cgroup::apply_limits(&cgroup, &spec.limits).map_err(cleanup)?;
    }
    let cgroup_fd = fs::File::open(&cgroup).map_err(cleanup)?;
    let host = super::hostopts::parse(spec.opts).map_err(cleanup)?;
    if spec.flags & run_flags::NET_HOST != 0 && host.sysctls.iter().any(|s| s.ns == super::hostopts::Ns::Net) {
        return Err(cleanup(io::Error::new(io::ErrorKind::InvalidInput, "net.* sysctls are not allowed in the host network namespace")));
    }
    // `--device` nodes are looked up now, so a missing one fails the start.
    let (nodes, device_rules) = super::hostopts::resolve_devices(&host).map_err(cleanup)?;
    // Deny block-device read/write (Docker's default device cgroup). If the
    // kernel cannot attach the filter, CAP_MKNOD is dropped below instead.
    let device_filtered = spec.flags & run_flags::PRIVILEGED != 0
        || match super::devices::restrict(cgroup_fd.as_raw_fd(), &device_rules) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("kiri: device filter unavailable, dropping CAP_MKNOD: {e}");
                false
            }
        };

    let mut prepared = prepare(spec.argv, spec.env, spec.cwd, spec.user, Some(spec.hostname), spec.flags, spec.caps).map_err(cleanup)?;
    if !device_filtered {
        prepared.caps &= !(1 << super::devices::CAP_MKNOD);
    }
    if spec.limits.shm_size > 0 {
        prepared.shm_size = spec.limits.shm_size;
    }
    prepared.domainname = net_opts::values(spec.net.opts, net_opts::DOMAINNAME).next().filter(|d| !d.is_empty()).map(|d| d.as_bytes().to_vec());
    let mut netns = 0;
    if let Some(target) = net_opts::values(spec.net.opts, net_opts::NETNS).next() {
        let pid = net.join_pid.ok_or_else(|| cleanup(io::Error::other(format!("cannot join network of a non running container: {target}"))))?;
        let joined = fs::File::open(format!("/proc/{pid}/ns/net")).map_err(cleanup)?;
        netns = netns_inode(&joined);
        prepared.net_join = Some(joined);
    }
    prepared.host = HostApplied { rlimits: host.rlimits.clone(), groups: host.groups.clone(), nodes };
    // Small, short-lived files: kept in memory (/run is tmpfs), not on the data disk.
    let etc_dir = PathBuf::from("/run/kiri").join(spec.id);
    let embedded_dns = embedded_dns(spec, net.upstream_dns, net.dns.relay);
    let nameserver = embedded_dns.then(|| embedded_address(spec, net.dns.relay));
    prepared.etc_files = write_etc_files(spec, &etc_dir, net.upstream_dns, nameserver).map_err(cleanup)?;
    prepared.mounts = parse_mounts(spec.mounts).map_err(cleanup)?;
    let bridged = !spec.net.bridge.is_empty() && spec.flags & (run_flags::NET_HOST | run_flags::NET_NONE) == 0 && prepared.net_join.is_none();
    let mut go = None;
    // Sysctls are written from here once the namespaces exist: the child
    // waits for that as it does for its network.
    if bridged || !host.sysctls.is_empty() {
        let (r, w) = sys::pipe().map_err(cleanup)?;
        prepared.net_ready = Some(r);
        go = Some(w);
    }
    let (parent_io, child_io) = stdio(&prepared).map_err(cleanup)?;
    let rootfs_c = sys::cstr(&rootfs).map_err(cleanup)?;

    let mut flags = (libc::CLONE_NEWNS | libc::CLONE_NEWPID | libc::CLONE_NEWUTS | libc::CLONE_NEWIPC | libc::CLONE_NEWCGROUP)
        as u64
        | CLONE_INTO_CGROUP;
    if spec.flags & run_flags::NET_HOST == 0 && prepared.net_join.is_none() {
        flags |= libc::CLONE_NEWNET as u64;
    }
    let args = CloneArgs { flags, exit_signal: libc::SIGCHLD as u64, cgroup: cgroup_fd.as_raw_fd() as u64, ..Default::default() };

    // SAFETY: the agent is single-threaded, as clone3 without CLONE_VM needs.
    let pid = unsafe { sys::clone3(&args) }.map_err(cleanup)?;
    if pid == 0 {
        container_child(&prepared, &rootfs_c, &child_io);
    }
    if flags & libc::CLONE_NEWNET as u64 != 0 {
        netns = fs::File::open(format!("/proc/{pid}/ns/net")).map_or(0, |f| netns_inode(&f));
    }
    drop(cgroup_fd);
    let kill_child = |e: io::Error| {
        // SAFETY: killing the child we just cloned.
        unsafe { libc::kill(pid, libc::SIGKILL) };
        e
    };
    let mut veths = Vec::new();
    // Not bridged, but with sysctls to write: no network to wait for.
    if let Some(go) = go.take_if(|_| !bridged) {
        drop(prepared.net_ready.take());
        super::hostopts::apply_sysctls(pid, &host.sysctls).map_err(kill_child)?;
        // SAFETY: one byte to a pipe we own: "go on".
        unsafe { libc::write(go.as_raw_fd(), b"g".as_ptr().cast(), 1) };
    }
    if let Some(go) = go {
        drop(prepared.net_ready.take());
        // Named after the container, plus a per-start counter: after a
        // restart the previous veth may still be on its way out with the old
        // network namespace. 15 characters at most, the kernel's limit.
        static STARTS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = STARTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) & 0xffff;
        let host_if = format!("kv{}{n:04x}", &spec.id[..spec.id.len().min(9)]);
        let mac = net_opts::values(spec.net.opts, net_opts::MAC).next().and_then(net::parse_mac);
        let v6 = net::Ip6::from_opts(spec.net.opts);
        let plug = net::Plug::from_opts(spec.net.opts);
        net::attach_container(net.bridges, spec.net.bridge, &host_if, pid, spec.net.ip, spec.net.prefix, spec.net.gateway, mac, v6, plug)
            .map_err(|e| kill_child(io::Error::new(e.kind(), format!("network: {e}"))))?;
        veths.push(host_if);
        if embedded_dns {
            // An IPv6-only network's bridge has no IPv4 address to serve on,
            // and its containers no A records.
            if !plug.no_ipv4 {
                net.dns.serve(spec.net.bridge, spec.net.gateway).map_err(|e| kill_child(io::Error::new(e.kind(), format!("dns: {e}"))))?;
                net.dns.register(spec.net.bridge, spec.net.dns_names.iter(), spec.net.ip);
            }
            // 127.0.0.11 in the container, what its resolv.conf names.
            std::fs::File::open(format!("/proc/{pid}/ns/net"))
                .and_then(|ns| net.dns.serve_in(&ns, pid, spec.net.ip, spec.net.bridge))
                .map_err(|e| kill_child(io::Error::new(e.kind(), format!("dns: {e}"))))?;
            if let Some(v) = v6 {
                net.dns.register6(spec.net.bridge, spec.net.dns_names.iter(), spec.net.ip, v.ip);
            }
            // `--dns`: names the network does not know go to these instead.
            let servers: Vec<[u8; 4]> = net_opts::values(spec.net.opts, net_opts::NAMESERVER).filter_map(net::parse_ip).collect();
            net.dns.set_upstream(spec.net.ip, servers);
        }
        super::hostopts::apply_sysctls(pid, &host.sysctls).map_err(kill_child)?;
        // SAFETY: one byte to a pipe we own: "your network is ready".
        unsafe { libc::write(go.as_raw_fd(), b"g".as_ptr().cast(), 1) };
    }
    let (stdin, stdout, stderr) = finish_stdio(parent_io, child_io).map_err(kill_child)?;
    Ok(Process {
        id: spec.id.to_owned(),
        container: None,
        pid,
        stdout,
        stderr,
        stdin,
        tty: prepared.tty,
        dns_entries: (bridged && embedded_dns).then(|| (spec.net.bridge.to_owned(), spec.net.ip)).into_iter().collect(),
        veths,
        netns,
        teardown: Some(Teardown { base, rootfs, cgroup, remove }),
    })
}

/// Mounts a container's root: its image layers under its writable layer,
/// at `<containers>/<id>/rootfs`. Returns that path.
pub fn mount_rootfs(paths: &Paths, id: &str, image_layers: StrList<'_>) -> io::Result<PathBuf> {
    valid_id(id)?;
    let base = paths.containers.join(id);
    let (upper, work, rootfs) = (base.join("upper"), base.join("work"), base.join("rootfs"));
    for d in [&upper, &work, &rootfs] {
        fs::create_dir_all(d)?;
    }
    // Lower dirs are relative to the agent's working directory (the layer
    // store), which keeps the option string short enough for many layers.
    let mut lower = String::with_capacity(image_layers.len() * 65);
    let mut seen = Vec::new();
    for digest in image_layers.iter() {
        if !layers::present(&paths.layers, digest) {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("layer {digest} is not unpacked")));
        }
        // An image can hold the same layer twice (often the empty one,
        // sha256:4f4fb700…), and the store keeps it once: overlayfs refuses
        // a directory listed twice, with ELOOP. The layers come top first,
        // and the topmost copy alone gives the same tree, so later ones go.
        if seen.contains(&digest) {
            continue;
        }
        seen.push(digest);
        if !lower.is_empty() {
            lower.push(':');
        }
        lower.push_str(layers::hex(digest)?);
    }
    let opts = format!("lowerdir={lower},upperdir={},workdir={}", upper.display(), work.display());
    sys::mount(c"overlay", &sys::cstr(&rootfs)?, c"overlay", 0, Some(&CString::new(opts)?))?;
    Ok(rootfs)
}

/// Starts a process inside a running container: same namespaces, same cgroup.
pub fn exec(spec: &ExecSpec<'_>, target_pid: libc::pid_t, paths: &Paths) -> io::Result<Process> {
    if spec.argv.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "no command to run"));
    }
    valid_id(spec.exec_id)?;
    let ns = |name: &str| fs::File::open(format!("/proc/{target_pid}/ns/{name}"));
    let target_pid_ns = ns("pid")?;
    let namespaces = [
        (ns("net")?, libc::CLONE_NEWNET),
        (ns("uts")?, libc::CLONE_NEWUTS),
        (ns("ipc")?, libc::CLONE_NEWIPC),
        (ns("cgroup")?, libc::CLONE_NEWCGROUP),
        // Last: entering the mount namespace also moves the root.
        (ns("mnt")?, libc::CLONE_NEWNS),
    ];
    let own_pid_ns = fs::File::open("/proc/self/ns/pid")?;
    let cgroup_fd = fs::File::open(paths.cgroup.join(spec.container_id))?;

    let mut prepared = prepare(spec.argv, spec.env, spec.cwd, spec.user, None, spec.flags, spec.caps)?;
    // The container's ulimits and groups apply to its execs too, as with runc.
    let host = super::hostopts::parse(spec.opts)?;
    prepared.host = HostApplied { rlimits: host.rlimits, groups: host.groups, nodes: Vec::new() };
    let (parent_io, child_io) = stdio(&prepared)?;
    let args = CloneArgs {
        flags: CLONE_INTO_CGROUP,
        exit_signal: libc::SIGCHLD as u64,
        cgroup: cgroup_fd.as_raw_fd() as u64,
        ..Default::default()
    };

    // setns(CLONE_NEWPID) only changes where the agent's *children* go, so
    // the exec is cloned straight into the container's PID namespace and
    // stays the agent's direct child. The agent's own namespace is put back
    // right after.
    sys::setns(target_pid_ns.as_raw_fd(), libc::CLONE_NEWPID)?;
    // SAFETY: the agent is single-threaded, as clone3 without CLONE_VM needs.
    let mut cloned = unsafe { sys::clone3(&args) };
    // A container that manages its own cgroups (Docker in Docker, BuildKit)
    // moves its processes into sub-cgroups, after which its own cgroup takes
    // no processes. Join the one its init lives in, as runc does.
    let init_cgroup;
    if cloned.as_ref().is_err_and(|e| e.kind() == io::ErrorKind::ResourceBusy) {
        let rel = fs::read_to_string(format!("/proc/{target_pid}/cgroup")).unwrap_or_default();
        let rel = rel.lines().find_map(|l| l.strip_prefix("0::")).unwrap_or("").trim_start_matches('/').to_owned();
        init_cgroup = fs::File::open(std::path::Path::new("/sys/fs/cgroup").join(&rel));
        match &init_cgroup {
            Ok(fd) => {
                let retry = CloneArgs { cgroup: fd.as_raw_fd() as u64, ..args };
                // SAFETY: as above.
                cloned = unsafe { sys::clone3(&retry) }.map_err(|e| io::Error::new(e.kind(), format!("joining cgroup {rel}: {e}")));
            }
            Err(e) => cloned = Err(io::Error::new(e.kind(), format!("cgroup {rel}: {e}"))),
        }
    }
    if let Ok(0) = cloned {
        exec_child(&prepared, &namespaces, &child_io);
    }
    let restored = sys::setns(own_pid_ns.as_raw_fd(), libc::CLONE_NEWPID);
    let pid = cloned?;
    restored?;
    drop(cgroup_fd);
    let (stdin, stdout, stderr) = finish_stdio(parent_io, child_io).map_err(|e| {
        // SAFETY: killing the child we just cloned.
        unsafe { libc::kill(pid, libc::SIGKILL) };
        e
    })?;
    Ok(Process {
        id: spec.exec_id.to_owned(),
        container: Some(spec.container_id.to_owned()),
        pid,
        stdout,
        stderr,
        stdin,
        tty: prepared.tty,
        dns_entries: Vec::new(),
        veths: Vec::new(),
        netns: 0,
        teardown: None,
    })
}

/// Unmounts a container and, for `--rm`, deletes its writable layer.
/// Runs after the exit has been reported, off the critical path. Execs have
/// nothing to undo.
/// Whether the container's processes were killed for memory. Read before
/// `teardown`, which removes the cgroup.
pub fn oom_killed(p: &Process) -> bool {
    p.teardown.as_ref().is_some_and(|t| super::cgroup::oom_killed(&t.cgroup))
}

/// `netns_shared`: another live process still uses the container's network
/// namespace (a container started with `--network container:`). Only then
/// are its veths deleted here, or they would live on with that namespace
/// and keep its address. Otherwise the kernel deletes them as the namespace
/// goes, off the agent's loop: deleting one here waits out RCU, 15-40 ms of
/// netlink that every request after the exit queued behind.
pub fn teardown(p: &Process, netns_shared: bool) {
    if netns_shared {
        for v in &p.veths {
            let _ = net::detach(v);
        }
    }
    if let Some(t) = &p.teardown {
        undo_now(t, t.remove);
    }
}

fn netns_inode(f: &fs::File) -> u64 {
    use std::os::unix::fs::MetadataExt;
    f.metadata().map_or(0, |m| m.ino())
}

/// Removes a cgroup and every cgroup below it, deepest first. They are
/// empty once the container's processes are gone; rmdir is all it takes.
fn remove_cgroup(dir: &std::path::Path) {
    if let Ok(entries) = fs::read_dir(dir) {
        for e in entries.flatten() {
            if e.file_type().is_ok_and(|t| t.is_dir()) {
                remove_cgroup(&e.path());
            }
        }
    }
    let _ = fs::remove_dir(dir);
}

fn undo_now(t: &Teardown, remove_files: bool) {
    let _ = sys::umount_lazy(&t.rootfs);
    if let Some(id) = t.base.file_name() {
        let _ = fs::remove_dir_all(PathBuf::from("/run/kiri").join(id));
    }
    remove_cgroup(&t.cgroup);
    if remove_files {
        let _ = fs::remove_dir_all(&t.base);
    }
}

/// Deletes a stopped container's directory (its writable layer).
/// Deletes a named volume's data.
pub fn remove_volume(name: &str) {
    // Docker's rule for volume names: [a-zA-Z0-9][a-zA-Z0-9_.-]*
    let valid = name.bytes().next().is_some_and(|b| b.is_ascii_alphanumeric())
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b));
    if valid {
        // An NFS export or a host folder mounted there is not the volume's
        // to delete: unmount it first, and keep everything if that fails.
        let data = format!("/var/lib/kiri/volumes/{name}/_data");
        if !super::volmount::release(&data) {
            eprintln!("kiri-init: volume {name}: {data} is still mounted, not deleting it");
            return;
        }
        let _ = fs::remove_dir_all(PathBuf::from("/var/lib/kiri/volumes").join(name));
    }
}

pub fn remove(paths: &Paths, id: &str) {
    if valid_id(id).is_ok() {
        let base = paths.containers.join(id);
        let _ = sys::umount_lazy(&base.join("rootfs"));
        let _ = fs::remove_dir_all(base);
    }
}

/// Linux host: what a previous agent that died without cleaning up (killed,
/// or its machine's daemon killed with it) left of this data directory's
/// containers under `cgroup`: their processes are killed, their mounts,
/// runtime files and cgroups removed, and a veth something else kept alive
/// deleted. Only containers of this data directory are touched: the
/// cgroup and the network namespace may hold another engine's too.
pub fn sweep(paths: &Paths, cgroup: &std::path::Path) {
    let Ok(entries) = fs::read_dir(&paths.containers) else { return };
    let ids: Vec<String> = entries.flatten().filter_map(|e| e.file_name().into_string().ok()).filter(|id| valid_id(id).is_ok() && id.len() >= 12).collect();
    for id in &ids {
        let cg = cgroup.join(id);
        if !cg.is_dir() {
            continue;
        }
        kill_cgroup(&cg);
        let _ = sys::umount_lazy(&paths.containers.join(id).join("rootfs"));
        let _ = fs::remove_dir_all(PathBuf::from("/run/kiri").join(id));
        remove_cgroup(&cg);
    }
    // Host-side veths are kv<9 characters of the id><4 hex digits>.
    let Ok(links) = fs::read_dir("/sys/class/net") else { return };
    for link in links.flatten() {
        let Ok(name) = link.file_name().into_string() else { continue };
        let ours = name.strip_prefix("kv").and_then(|rest| rest.get(..9)).is_some_and(|p| ids.iter().any(|id| id.starts_with(p)));
        if ours {
            let _ = net::detach(&name);
        }
    }
}

/// Kills every process in a cgroup and below it, and waits (briefly) until
/// it is empty.
fn kill_cgroup(dir: &std::path::Path) {
    // cgroup.kill is Linux 5.14+; before it, each process, deepest first.
    if fs::write(dir.join("cgroup.kill"), "1").is_err() {
        fn each(dir: &std::path::Path) {
            if let Ok(entries) = fs::read_dir(dir) {
                for e in entries.flatten().filter(|e| e.file_type().is_ok_and(|t| t.is_dir())) {
                    each(&e.path());
                }
            }
            for pid in fs::read_to_string(dir.join("cgroup.procs")).unwrap_or_default().lines().filter_map(|l| l.parse::<libc::pid_t>().ok()) {
                // SAFETY: SIGKILL to a process of a container cgroup we own.
                unsafe { libc::kill(pid, libc::SIGKILL) };
            }
        }
        each(dir);
    }
    for _ in 0..300 {
        if fs::read_to_string(dir.join("cgroup.events")).map_or(true, |e| e.lines().any(|l| l == "populated 0")) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Deletes every container directory not in `keep` and not in `running`.
pub fn retain(paths: &Paths, keep: StrList<'_>, running: &[&str]) {
    let Ok(entries) = fs::read_dir(&paths.containers) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !keep.iter().any(|k| k == name) && !running.contains(&name) {
            remove(paths, name);
        }
    }
}

fn valid_id(id: &str) -> io::Result<()> {
    if id.is_empty() || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "bad id"));
    }
    Ok(())
}

fn prepare(argv: StrList<'_>, env: StrList<'_>, cwd: &str, user: &str, hostname: Option<&str>, flags: u32, caps: u64) -> io::Result<Prepared> {
    let c = |s: &str| CString::new(s).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "argument contains NUL"));
    let argv = argv.iter().map(c).collect::<io::Result<Vec<_>>>()?;
    let mut envp = env.iter().map(c).collect::<io::Result<Vec<_>>>()?;
    let path = env.iter().find_map(|e| e.strip_prefix("PATH=")).unwrap_or(DEFAULT_PATH);
    if !env.iter().any(|e| e.starts_with("PATH=")) {
        envp.push(c(&format!("PATH={DEFAULT_PATH}"))?);
    }
    if let Some(h) = hostname {
        if !env.iter().any(|e| e.starts_with("HOSTNAME=")) {
            envp.push(c(&format!("HOSTNAME={h}"))?);
        }
    }
    let tty = flags & run_flags::TTY != 0;
    if tty && !env.iter().any(|e| e.starts_with("TERM=")) {
        envp.push(c("TERM=xterm")?);
    }
    Ok(Prepared {
        argv,
        envp,
        path_dirs: path.split(':').filter(|d| !d.is_empty()).map(|d| d.as_bytes().to_vec()).collect(),
        hostname: hostname.map(|h| h.as_bytes().to_vec()),
        domainname: None,
        net_join: None,
        etc_files: Vec::new(),
        mounts: Vec::new(),
        net_ready: None,
        cwd: c(if cwd.is_empty() { "/" } else { cwd })?,
        user: user.to_owned(),
        net_none: flags & run_flags::NET_NONE != 0,
        tty,
        stdin: flags & run_flags::STDIN != 0,
        privileged: flags & run_flags::PRIVILEGED != 0,
        caps: if caps == 0 { default_caps() } else { caps },
        seccomp: flags & (run_flags::PRIVILEGED | run_flags::NO_SECCOMP) == 0,
        no_new_privs: flags & run_flags::NO_NEW_PRIVS != 0,
        readonly_rootfs: flags & run_flags::READONLY_ROOTFS != 0,
        shm_size: 64 << 20,
        init: flags & run_flags::INIT != 0,
        host: HostApplied::default(),
    })
}

/// Writes the container's /etc/hosts, /etc/resolv.conf and /etc/hostname
/// beside its writable layer; the child binds them over the image's files.
fn write_etc_files(spec: &RunSpec<'_>, base: &std::path::Path, upstream: &[[u8; 4]], embedded: Option<[u8; 4]>) -> io::Result<Vec<(CString, CString)>> {
    fs::create_dir_all(base)?;
    // `--network container:<id>` shares that container's files, as Docker does.
    if let Some(target) = net_opts::values(spec.net.opts, net_opts::NETNS).next() {
        valid_id(target)?;
        let theirs = PathBuf::from("/run/kiri").join(target);
        let mut out = Vec::new();
        for name in ["hosts", "resolv.conf", "hostname"] {
            let src = theirs.join(name);
            fs::metadata(&src).map_err(|e| io::Error::new(e.kind(), format!("network container {target}: {name}: {e}")))?;
            out.push((sys::cstr(&src)?, CString::new(format!("etc/{name}")).expect("no NUL")));
        }
        return Ok(out);
    }
    let domain = net_opts::values(spec.net.opts, net_opts::DOMAINNAME).next().filter(|d| !d.is_empty());
    let mut hosts = String::from("127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n");
    // On an IPv6-only network the IPv4 address is Kiri's own name for it.
    let v4 = spec.net.ip != [0; 4] && !net::Plug::from_opts(spec.net.opts).no_ipv4;
    if v4 {
        match domain {
            Some(d) => hosts.push_str(&format!("{}\t{}.{d} {}\n", net::ip_str(spec.net.ip), spec.hostname, spec.hostname)),
            None => hosts.push_str(&format!("{}\t{}\n", net::ip_str(spec.net.ip), spec.hostname)),
        }
    }
    if spec.net.ip != [0; 4] {
        // On an IPv6 network the IPv6 address names it too, as Docker writes it.
        if let Some(v) = net::Ip6::from_opts(spec.net.opts) {
            match domain {
                Some(d) => hosts.push_str(&format!("{}\t{}.{d} {}\n", net::ip6_str(v.ip), spec.hostname, spec.hostname)),
                None => hosts.push_str(&format!("{}\t{}\n", net::ip6_str(v.ip), spec.hostname)),
            }
        }
    }
    // On a Linux host there is no loopback redirect; host.docker.internal
    // points at the bridge gateway, like Docker's host-gateway, so it reaches
    // host services on 0.0.0.0 but not ones bound only to 127.0.0.1.
    let host_addr = net::ip_str(net::HOST_ADDR);
    let gateway = net::ip_str(spec.net.gateway);
    let map_host = super::transport::is_host_process() && spec.net.gateway != [0; 4];
    for line in spec.net.extra_hosts.iter() {
        if map_host {
            hosts.push_str(&line.replace(&host_addr, &gateway));
        } else {
            hosts.push_str(line);
        }
        hosts.push('\n');
    }
    let resolv = resolv_conf(spec, upstream, embedded);
    let files = [("hosts", hosts), ("resolv.conf", resolv), ("hostname", format!("{}\n", spec.hostname))];
    let mut out = Vec::with_capacity(files.len());
    for (name, text) in files {
        let src = base.join(name);
        fs::write(&src, text)?;
        out.push((sys::cstr(&src)?, CString::new(format!("etc/{name}")).expect("no NUL")));
    }
    Ok(out)
}

/// The container's resolv.conf. On a user-defined network the embedded DNS
/// at 127.0.0.11 is the only nameserver, as with Docker (it forwards to
/// `--dns` servers); elsewhere `--dns` servers replace the VM's.
/// `--dns-search` and `--dns-option` apply either way.
/// Whether the container's DNS goes to the agent's resolver (127.0.0.11):
/// on a user network, as Docker's embedded DNS; and on the default bridge
/// when the host's resolver is one of its own addresses (net::local_resolver)
/// or, in the VM (`relay`), so that lookups the Mac's gateway resolver fails
/// can go through the host (dns.rs), unless the container names its own
/// servers (`--dns`).
fn embedded_dns(spec: &RunSpec<'_>, upstream: &[[u8; 4]], relay: bool) -> bool {
    if spec.flags & run_flags::DNS_GATEWAY != 0 {
        return true;
    }
    let bridged = !spec.net.bridge.is_empty()
        && spec.flags & (run_flags::NET_HOST | run_flags::NET_NONE) == 0
        && net_opts::values(spec.net.opts, net_opts::NETNS).next().is_none();
    bridged && net_opts::values(spec.net.opts, net_opts::NAMESERVER).next().is_none() && (relay || net::local_resolver(upstream))
}

/// The address a container that uses the agent's resolver names in its
/// resolv.conf: 127.0.0.11 on a user network, as Docker's embedded DNS;
/// on the default bridge in the VM, the bridge's gateway, where the agent
/// answers too. Not a loopback address there, because what runs builds
/// in a container takes loopback nameservers out of the copy it gives
/// each step and puts Google's in their place (BuildKit does, for every
/// step that isn't `--network=host`): those steps would ask 8.8.8.8, not
/// the Mac's resolver, and VPN and company names would stop resolving.
fn embedded_address(spec: &RunSpec<'_>, relay: bool) -> [u8; 4] {
    if relay && spec.flags & run_flags::DNS_GATEWAY == 0 && spec.net.gateway != [0; 4] { spec.net.gateway } else { super::dns::EMBEDDED }
}

fn resolv_conf(spec: &RunSpec<'_>, upstream: &[[u8; 4]], embedded: Option<[u8; 4]>) -> String {
    let mut resolv = String::new();
    let mut servers = net_opts::values(spec.net.opts, net_opts::NAMESERVER).peekable();
    if let Some(ns) = embedded {
        resolv.push_str(&format!("nameserver {}\n", net::ip_str(ns)));
    } else if servers.peek().is_some() {
        for d in servers {
            resolv.push_str(&format!("nameserver {d}\n"));
        }
    } else if upstream.is_empty() {
        // The VM has no lease yet; a public resolver beats none.
        resolv.push_str("nameserver 1.1.1.1\n");
    } else {
        for d in upstream {
            resolv.push_str(&format!("nameserver {}\n", net::ip_str(*d)));
        }
    }
    // `--dns-search .` means no search domains.
    let search: Vec<&str> = net_opts::values(spec.net.opts, net_opts::SEARCH).filter(|s| *s != ".").collect();
    if !search.is_empty() {
        resolv.push_str(&format!("search {}\n", search.join(" ")));
    }
    let options: Vec<&str> = net_opts::values(spec.net.opts, net_opts::OPTION).collect();
    if !options.is_empty() {
        resolv.push_str(&format!("options {}\n", options.join(" ")));
    }
    resolv
}

/// One mount, resolved before clone.
struct Mount {
    /// A guest path, or None for a tmpfs.
    source: Option<CString>,
    /// Relative to the container root (the child mounts before pivot_root).
    target: CString,
    ro: bool,
    dir: bool,
    /// Fill an empty volume from the image's files at the target, as
    /// Docker does the first time a volume is used.
    copy_up: bool,
    /// tmpfs options (`size=64m,mode=1777`).
    tmpfs_opts: Option<CString>,
    /// tmpfs mount flags (`noexec`, `nosuid`, `nodev`, `ro`).
    tmpfs_flags: libc::c_ulong,
}

/// `source\ttarget\tflags` entries. Flags: `ro`, `create` (make the source
/// directory), `copyup`, `tmpfs` (source ignored) and tmpfs options and
/// flags (`noexec`, `exec`, `nosuid`, …; later ones win). A volume with
/// driver options adds `\ttype\tdevice\tflags(hex)\tdata`, and is mounted
/// at its source first.
fn parse_mounts(list: StrList<'_>) -> io::Result<Vec<Mount>> {
    let mut out = Vec::with_capacity(list.len());
    for m in list.iter() {
        let mut parts = m.split('\t');
        let (Some(src), Some(target)) = (parts.next(), parts.next()) else {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("bad mount {m}")));
        };
        let flags: Vec<&str> = parts.next().unwrap_or("").split(',').filter(|f| !f.is_empty()).collect();
        if !target.starts_with('/') || target.split('/').any(|c| c == "..") {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("bad mount target {target}")));
        }
        let rel = target.trim_start_matches('/');
        let target = CString::new(if rel.is_empty() { "." } else { rel })?;
        let ro = flags.contains(&"ro");
        if flags.contains(&"tmpfs") {
            let opts: Vec<&str> = flags.iter().copied().filter(|f| f.contains('=')).collect();
            let opts = if opts.is_empty() { "mode=1777".to_owned() } else { opts.join(",") };
            let tmpfs_flags = tmpfs_flags(&flags);
            let ro = tmpfs_flags & libc::MS_RDONLY != 0;
            out.push(Mount { source: None, target, ro, dir: true, copy_up: false, tmpfs_opts: Some(CString::new(opts)?), tmpfs_flags });
            continue;
        }
        // A missing source becomes a directory, as Docker does; one that is
        // there already (a file such as /etc/hostname, too) is used as it is.
        if flags.contains(&"create") && fs::symlink_metadata(src).is_err() {
            fs::create_dir_all(src)?;
        }
        if let (Some(kind), Some(device), Some(mflags), data) = (parts.next().filter(|k| !k.is_empty()), parts.next(), parts.next(), parts.next()) {
            super::volmount::ensure(src, kind, device, mflags, data.unwrap_or(""))?;
        }
        let meta = fs::metadata(src).map_err(|e| io::Error::new(e.kind(), format!("mount source {src}: {e}")))?;
        out.push(Mount {
            source: Some(sys::cstr(src)?),
            target,
            ro,
            dir: meta.is_dir(),
            copy_up: flags.contains(&"copyup"),
            tmpfs_opts: None,
            tmpfs_flags: 0,
        });
    }
    Ok(out)
}

/// A tmpfs's mount flags from its flag words, in order, so `exec` after
/// Docker's default `noexec` wins.
fn tmpfs_flags(words: &[&str]) -> libc::c_ulong {
    let mut f = 0;
    for w in words {
        match *w {
            "ro" => f |= libc::MS_RDONLY,
            "rw" => f &= !libc::MS_RDONLY,
            "noexec" => f |= libc::MS_NOEXEC,
            "exec" => f &= !libc::MS_NOEXEC,
            "nosuid" => f |= libc::MS_NOSUID,
            "suid" => f &= !libc::MS_NOSUID,
            "nodev" => f |= libc::MS_NODEV,
            "dev" => f &= !libc::MS_NODEV,
            "noatime" => f |= libc::MS_NOATIME,
            "atime" => f &= !libc::MS_NOATIME,
            _ => {}
        }
    }
    f
}

/// Mounts one entry into the container (relative target, before
/// pivot_root), creating the target as a directory or an empty file. The
/// target is looked up inside `root`, so an image's symlink cannot point
/// the mount, the copy-up or the new directory outside the container.
fn apply_mount(root: BorrowedFd<'_>, m: &Mount) -> io::Result<()> {
    let t = Path::new(std::ffi::OsStr::from_bytes(m.target.to_bytes()));
    if let Some(opts) = &m.tmpfs_opts {
        let at = inroot::open(root, t, Last::Dir)?;
        return sys::mount(c"tmpfs", &at.proc_path(), c"tmpfs", m.tmpfs_flags, Some(opts));
    }
    let src = m.source.as_ref().expect("bind mounts have a source");
    let at = inroot::open(root, t, if m.dir { Last::Dir } else { Last::File })?;
    let target = at.proc_path();
    if m.dir && m.copy_up && !at.created {
        let src_path = Path::new(std::ffi::OsStr::from_bytes(src.to_bytes()));
        let image_path = Path::new(std::ffi::OsStr::from_bytes(target.to_bytes()));
        let empty = fs::read_dir(src_path).map(|mut d| d.next().is_none()).unwrap_or(false);
        if empty && image_path.is_dir() {
            copy_tree(image_path, src_path)?;
        }
    }
    sys::mount(src, &target, c"none", libc::MS_BIND | libc::MS_REC, None)?;
    if m.ro {
        // The descriptor is the directory under the new mount; look again
        // to reach the mount itself.
        let at = inroot::open(root, t, Last::Exists)?;
        sys::mount(c"none", &at.proc_path(), c"none", libc::MS_BIND | libc::MS_REMOUNT | libc::MS_RDONLY | libc::MS_REC, None)?;
    }
    Ok(())
}

/// Copies a directory tree keeping modes, owners and symlinks: what Docker
/// does to fill a new volume from the image. `from` may be a
/// `/proc/self/fd/N` path to the image's directory (see `apply_mount`): its
/// own owner and mode are read through the link, never off the link itself,
/// which is root's with mode 000.
fn copy_tree(from: &std::path::Path, to: &std::path::Path) -> io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let meta = fs::metadata(from)?;
    // Owner first, then mode: a chown clears set-user-ID and set-group-ID.
    let keep = |p: &std::path::Path, m: &fs::Metadata| -> io::Result<()> {
        let c = sys::cstr(p)?;
        // SAFETY: valid NUL-terminated path; lchown does not follow links.
        unsafe { libc::lchown(c.as_ptr(), m.uid(), m.gid()) };
        if !m.file_type().is_symlink() {
            fs::set_permissions(p, fs::Permissions::from_mode(m.mode() & 0o7777))?;
        }
        Ok(())
    };
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let (src, dst) = (entry.path(), to.join(entry.file_name()));
        let m = fs::symlink_metadata(&src)?;
        if m.file_type().is_symlink() {
            std::os::unix::fs::symlink(fs::read_link(&src)?, &dst)?;
        } else if m.is_dir() {
            fs::create_dir_all(&dst)?;
            copy_tree(&src, &dst)?;
            continue; // copy_tree set its owner and mode
        } else if m.is_file() {
            fs::copy(&src, &dst)?;
        } else {
            continue; // devices, fifos and sockets are not copied
        }
        keep(&dst, &m)?;
    }
    keep(to, &meta)
}

/// Binds `src` over `target` (relative to the container root, looked up
/// inside it), creating the target (and /etc) first if the image does not
/// have it.
fn bind_file(root: BorrowedFd<'_>, src: &CStr, target: &CStr) -> io::Result<()> {
    // Only create the target when the image lacks it: opening an existing
    // lower-layer file for writing would copy it up into the writable layer,
    // three times per container, for nothing. A bind mount sits over a
    // lower-layer file just fine, and an O_PATH lookup copies nothing.
    let at = inroot::open(root, Path::new(std::ffi::OsStr::from_bytes(target.to_bytes())), Last::File)?;
    sys::mount(src, &at.proc_path(), c"none", libc::MS_BIND, None)
}

/// Where the child reports a failure before it has execed: the stderr pipe,
/// or the pty socket (the agent reads text there instead of a descriptor).
fn report_fd(io: &ChildStdio) -> RawFd {
    match io {
        ChildStdio::Tty { sock } => sock.as_raw_fd(),
        ChildStdio::Pipes { stderr, .. } => stderr.as_raw_fd(),
    }
}

/// Runs in the cloned child: PID 1 of the container, in its own namespaces
/// and cgroup. Never returns: it execs or exits 127 with a message.
fn container_child(p: &Prepared, rootfs: &CStr, io: &ChildStdio) -> ! {
    let err = report_fd(io);
    let fail = |what: &str, e: io::Error| -> ! { child_fail(err, what, e) };

    // Mount events must not leak back into the agent's namespace.
    sys::remount(c"/", libc::MS_REC | libc::MS_PRIVATE).unwrap_or_else(|e| fail("make mounts private", e));
    if let Some(ns) = &p.net_join {
        sys::setns(ns.as_raw_fd(), libc::CLONE_NEWNET).unwrap_or_else(|e| fail("join network container", e));
    }
    // SAFETY: `rootfs` is a valid NUL-terminated path.
    if unsafe { libc::chdir(rootfs.as_ptr()) } != 0 {
        fail("enter rootfs", io::Error::last_os_error());
    }
    let root = inroot::open_root(Path::new(".")).unwrap_or_else(|e| fail("enter rootfs", e));
    let root = root.as_fd();
    mount_container_filesystems(p.privileged, p.shm_size).unwrap_or_else(|e| fail("mount", e));
    super::hostopts::make_nodes(root, &p.host.nodes).unwrap_or_else(|e| fail("devices", e));
    for (src, target) in &p.etc_files {
        bind_file(root, src, target).unwrap_or_else(|e| fail(&target.to_string_lossy(), e));
    }
    for m in &p.mounts {
        apply_mount(root, m).unwrap_or_else(|e| fail(&format!("mount {}", m.target.to_string_lossy()), e));
    }

    // pivot_root(".", ".") stacks the old root under the new one; detaching
    // it leaves only the container's filesystem visible.
    // SAFETY: both arguments are valid NUL-terminated strings.
    if unsafe { libc::syscall(libc::SYS_pivot_root, c".".as_ptr(), c".".as_ptr()) } != 0 {
        fail("pivot_root", io::Error::last_os_error());
    }
    // SAFETY: valid NUL-terminated path.
    unsafe {
        libc::umount2(c".".as_ptr(), libc::MNT_DETACH);
        libc::chdir(c"/".as_ptr());
    }
    if p.readonly_rootfs {
        // Only the root itself: volumes, tmpfs and /dev stay as they are.
        sys::mount(c"none", c"/", c"none", libc::MS_REMOUNT | libc::MS_BIND | libc::MS_RDONLY, None).unwrap_or_else(|e| fail("read-only root", e));
    }
    if let Some(h) = &p.hostname {
        sys::sethostname(h).unwrap_or_else(|e| fail("hostname", e));
    }
    if let Some(d) = &p.domainname {
        sys::setdomainname(d).unwrap_or_else(|e| fail("domainname", e));
    }
    if p.net_none {
        sys::loopback_up().unwrap_or_else(|e| fail("loopback", e));
    }
    if let Some(ready) = &p.net_ready {
        let mut b = [0u8; 1];
        // SAFETY: reading one byte into a local buffer.
        if unsafe { libc::read(ready.as_raw_fd(), b.as_mut_ptr().cast(), 1) } != 1 {
            fail("network", io::Error::other("the agent could not set up this container's network"));
        }
    }
    finish_child(p, io, true)
}

/// Runs in the cloned child of an exec: joins the container's remaining
/// namespaces (the PID namespace was set before clone), then execs.
fn exec_child(p: &Prepared, namespaces: &[(fs::File, libc::c_int)], io: &ChildStdio) -> ! {
    let err = report_fd(io);
    for (fd, kind) in namespaces {
        sys::setns(fd.as_raw_fd(), *kind).unwrap_or_else(|e| child_fail(err, "join container", e));
    }
    // SAFETY: valid NUL-terminated path.
    unsafe { libc::chdir(c"/".as_ptr()) };
    finish_child(p, io, false)
}

/// Binds the container's terminal over /dev/console; a privileged
/// container's devtmpfs has the VM's there, an unprivileged one has none.
fn bind_console(slave: &OwnedFd) {
    let Ok(pts) = fs::read_link(format!("/proc/self/fd/{}", slave.as_raw_fd())) else { return };
    let Ok(pts) = CString::new(pts.into_os_string().into_encoded_bytes()) else { return };
    if fs::symlink_metadata("/dev/console").is_err() {
        let _ = fs::File::create("/dev/console");
    }
    let _ = sys::mount(&pts, c"/dev/console", c"", libc::MS_BIND, None);
}

/// The part containers and execs share: standard streams, capabilities,
/// user, working directory, exec. `console`: a container's own start, whose
/// terminal also becomes its /dev/console (an exec's doesn't).
fn finish_child(p: &Prepared, io: &ChildStdio, console: bool) -> ! {
    let err = report_fd(io);
    let fail = |what: &str, e: io::Error| -> ! { child_fail(err, what, e) };

    match io {
        ChildStdio::Tty { sock } => {
            // The pty comes from the container's own /dev/pts, so `tty` and
            // job control see a terminal that exists inside the container.
            let (master, slave) = sys::open_pty().unwrap_or_else(|e| fail("terminal", e));
            sys::send_fd(sock.as_raw_fd(), master.as_raw_fd()).unwrap_or_else(|e| fail("terminal", e));
            drop(master);
            // As runc does: /dev/console is this terminal, not the VM's
            // console, so what an init such as systemd writes there reaches
            // `docker logs` (kind waits for its "Reached target" line).
            if console {
                bind_console(&slave);
            }
            // SAFETY: new session with the pty as its controlling terminal,
            // then the pty as all three standard streams.
            unsafe {
                libc::setsid();
                if libc::ioctl(slave.as_raw_fd(), libc::TIOCSCTTY as libc::Ioctl, 0) != 0
                    || libc::dup2(slave.as_raw_fd(), 0) < 0
                    || libc::dup2(slave.as_raw_fd(), 1) < 0
                    || libc::dup2(slave.as_raw_fd(), 2) < 0
                {
                    fail("terminal", io::Error::last_os_error());
                }
            }
        }
        ChildStdio::Pipes { stdin, stdout, stderr } => {
            // SAFETY: plain descriptor operations on descriptors we own.
            unsafe {
                let input = match stdin {
                    Some(fd) => fd.as_raw_fd(),
                    None => libc::open(c"/dev/null".as_ptr(), libc::O_RDONLY),
                };
                if input < 0 || libc::dup2(input, 0) < 0 || libc::dup2(stdout.as_raw_fd(), 1) < 0 || libc::dup2(stderr.as_raw_fd(), 2) < 0 {
                    fail("stdio", io::Error::last_os_error());
                }
            }
        }
    }

    // From here on stderr is the container's own (the pipe or the terminal),
    // which is where the user will see why a start failed.
    let fail = |what: &str, e: io::Error| -> ! { child_fail(2, what, e) };

    // While still root with every capability: raising a hard limit needs
    // CAP_SYS_RESOURCE.
    super::hostopts::set_rlimits(&p.host.rlimits).unwrap_or_else(|e| fail("ulimits", e));

    for cap in 0..=LAST_CAP {
        if !p.privileged && p.caps & (1 << cap) == 0 {
            // SAFETY: PR_CAPBSET_DROP takes a capability number; EINVAL for
            // capabilities this kernel does not know is fine to ignore.
            unsafe { libc::prctl(libc::PR_CAPBSET_DROP, cap as libc::c_ulong, 0, 0, 0) };
        }
    }

    if p.no_new_privs {
        // SAFETY: a plain prctl.
        if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
            fail("no-new-privileges", io::Error::last_os_error());
        }
    }
    // While still root with CAP_SYS_ADMIN, which installing a filter needs.
    // Without the default profile the host boundary stays: vsock is not
    // namespaced, and privileges inside the VM are no reason to reach the host.
    super::seccomp::install(p.seccomp.then_some(p.caps)).unwrap_or_else(|e| fail("seccomp", e));
    let (uid, gid) = resolve_user(&p.user).unwrap_or_else(|e| fail("user", e));
    // The user's /etc/group memberships and --group-add, as Docker gives them.
    let groups = super::hostopts::supplementary_groups(&p.user, uid, &p.host.groups).unwrap_or_else(|e| fail("groups", e));
    // SAFETY: plain id-setting syscalls; `groups` outlives setgroups.
    unsafe {
        if libc::setgroups(groups.len() as _, groups.as_ptr()) != 0 || libc::setgid(gid) != 0 || libc::setuid(uid) != 0 {
            fail("switch user", io::Error::last_os_error());
        }
        if libc::chdir(p.cwd.as_ptr()) != 0 {
            // Docker creates the whole working directory path, like mkdir -p.
            let _ = fs::create_dir_all(std::ffi::OsStr::from_encoded_bytes_unchecked(p.cwd.as_bytes()));
            if libc::chdir(p.cwd.as_ptr()) != 0 {
                fail("working directory", io::Error::last_os_error());
            }
        }
    }

    let exe = find_executable(p).unwrap_or_else(|| {
        fail(&p.argv[0].to_string_lossy(), io::Error::new(io::ErrorKind::NotFound, "executable file not found in $PATH"))
    });
    let mut argv: Vec<*const libc::c_char> = p.argv.iter().map(|a| a.as_ptr()).collect();
    argv.push(std::ptr::null());
    // Docker sets HOME from the user's passwd entry when the image and the
    // run do not: /root for root, else / if the user has no entry.
    let home = (!p.envp.iter().any(|e| e.as_bytes().starts_with(b"HOME="))).then(|| {
        let dir = home_of(uid).unwrap_or_else(|| if uid == 0 { "/root".to_owned() } else { "/".to_owned() });
        CString::new(format!("HOME={dir}")).unwrap_or_default()
    });
    let mut envp: Vec<*const libc::c_char> = p.envp.iter().map(|e| e.as_ptr()).collect();
    if let Some(h) = &home {
        envp.push(h.as_ptr());
    }
    envp.push(std::ptr::null());
    if console && p.init {
        super::pid1::run(&exe, argv.as_ptr(), envp.as_ptr(), p.tty);
    }
    // SAFETY: NULL-terminated arrays of valid C strings that outlive the call.
    unsafe { libc::execve(exe.as_ptr(), argv.as_ptr(), envp.as_ptr()) };
    fail("exec", io::Error::last_os_error())
}

/// The container's /proc, /sys, /dev (with the standard device nodes),
/// /dev/pts, /dev/shm, /dev/mqueue and a read-only cgroup view. Relative
/// paths: the current directory is the container root. A privileged
/// container gets the VM's devices and writable /sys and cgroups instead.
fn mount_container_filesystems(privileged: bool, shm_size: u64) -> io::Result<()> {
    let nosuid_nodev_noexec = libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC;
    let ro = if privileged { 0 } else { libc::MS_RDONLY };
    for d in [c"proc", c"sys", c"dev"] {
        sys::mkdir(d, 0o755);
        // Everything below is mounted by relative path, so these must be
        // the image's own directories, not symlinks out of it (runc refuses
        // the same).
        let path = Path::new(std::ffi::OsStr::from_bytes(d.to_bytes()));
        if !fs::symlink_metadata(path)?.is_dir() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("/{} in the image must be a directory", path.display())));
        }
    }
    sys::mount(c"proc", c"proc", c"proc", nosuid_nodev_noexec, None)?;
    sys::mount(c"sysfs", c"sys", c"sysfs", nosuid_nodev_noexec | ro, None)?;
    if privileged {
        sys::mount(c"devtmpfs", c"dev", c"devtmpfs", libc::MS_NOSUID, None)?;
    } else {
        sys::mount(c"tmpfs", c"dev", c"tmpfs", libc::MS_NOSUID, Some(c"mode=755,size=65536k"))?;
    }
    for (name, major, minor) in [
        (c"dev/null", 1, 3),
        (c"dev/zero", 1, 5),
        (c"dev/full", 1, 7),
        (c"dev/random", 1, 8),
        (c"dev/urandom", 1, 9),
        (c"dev/tty", 5, 0),
    ] {
        if privileged {
            // devtmpfs has them already.
            continue;
        }
        sys::mknod(name, libc::S_IFCHR | 0o666, major, minor)?;
        // mknod applies the umask; every user in the container needs these.
        // SAFETY: valid NUL-terminated path.
        unsafe { libc::chmod(name.as_ptr(), 0o666) };
    }
    for (target, link) in [
        (c"/proc/self/fd", c"dev/fd"),
        (c"/proc/self/fd/0", c"dev/stdin"),
        (c"/proc/self/fd/1", c"dev/stdout"),
        (c"/proc/self/fd/2", c"dev/stderr"),
        (c"pts/ptmx", c"dev/ptmx"),
    ] {
        // SAFETY: valid NUL-terminated strings; devtmpfs may have some.
        unsafe {
            libc::unlink(link.as_ptr());
            libc::symlink(target.as_ptr(), link.as_ptr())
        };
    }
    for d in [c"dev/pts", c"dev/shm", c"dev/mqueue"] {
        sys::mkdir(d, 0o755);
    }
    sys::mount(c"devpts", c"dev/pts", c"devpts", libc::MS_NOSUID | libc::MS_NOEXEC, Some(c"newinstance,ptmxmode=0666,mode=0620,gid=5"))?;
    let shm = CString::new(format!("mode=1777,size={}k", shm_size.div_ceil(1024))).expect("no NUL");
    sys::mount(c"shm", c"dev/shm", c"tmpfs", nosuid_nodev_noexec, Some(&shm))?;
    sys::mount(c"mqueue", c"dev/mqueue", c"mqueue", nosuid_nodev_noexec, None)?;
    sys::mkdir(c"sys/fs/cgroup", 0o755);
    sys::mount(c"cgroup2", c"sys/fs/cgroup", c"cgroup2", nosuid_nodev_noexec | ro, None)?;
    if !privileged {
        harden_proc()?;
    }
    Ok(())
}

/// Makes the dangerous parts of /proc read-only or hides them, as Docker's
/// default `readonlyPaths` and `maskedPaths` do. The container is real root
/// (no user namespace), and these sysctls and files are not namespaced, so
/// without this a container could set `kernel.core_pattern` or
/// `kernel.modprobe` and run code on the host. Paths are relative to the
/// container root (the current directory before pivot_root).
fn harden_proc() -> io::Result<()> {
    let ro = libc::MS_BIND | libc::MS_REMOUNT | libc::MS_RDONLY | libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC;
    // /proc/sys holds writable sysctls (core_pattern, modprobe, …): a path
    // to host code. This one must succeed.
    sys::mount(c"proc/sys", c"proc/sys", c"none", libc::MS_BIND, None)?;
    sys::remount(c"proc/sys", ro as libc::c_ulong)?;
    // The rest are best-effort: a path the guest kernel lacks is already
    // absent, so failing to cover it is not a hole.
    for p in [c"proc/sysrq-trigger", c"proc/irq", c"proc/bus", c"proc/fs"] {
        if sys::mount(p, p, c"none", libc::MS_BIND, None).is_ok() {
            let _ = sys::remount(p, ro as libc::c_ulong);
        }
    }
    // Hide files that leak host memory or kernel addresses, by covering them
    // with /dev/null (already created above).
    for p in [c"proc/kcore", c"proc/keys", c"proc/latency_stats", c"proc/timer_list", c"proc/timer_stats", c"proc/sched_debug", c"proc/interrupts"] {
        let _ = sys::mount(c"dev/null", p, c"none", libc::MS_BIND, None);
    }
    // Directories to hide, covered with an empty read-only tmpfs.
    for p in [c"proc/scsi", c"proc/acpi", c"proc/asound", c"sys/firmware", c"sys/devices/virtual/powercap"] {
        let _ = sys::mount(c"tmpfs", p, c"tmpfs", (libc::MS_RDONLY | libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC) as libc::c_ulong, Some(c"size=0k"));
    }
    Ok(())
}

/// `""`, `root`, `1000`, `1000:1000`, `redis`, `redis:redis`: looked up in
/// the container's /etc/passwd and /etc/group when not numeric.
/// The home directory /etc/passwd gives `uid`.
fn home_of(uid: u32) -> Option<String> {
    let text = fs::read_to_string("/etc/passwd").ok()?;
    text.lines().find_map(|l| {
        let f: Vec<&str> = l.split(':').collect();
        (f.len() >= 6 && f[2].parse() == Ok(uid)).then(|| f[5].to_owned()).filter(|h| !h.is_empty())
    })
}

fn resolve_user(user: &str) -> io::Result<(libc::uid_t, libc::gid_t)> {
    if user.is_empty() {
        return Ok((0, 0));
    }
    let (u, g) = user.split_once(':').map_or((user, None), |(u, g)| (u, Some(g)));
    let lookup = |file: &str, name: &str| -> Option<(u32, u32)> {
        let text = fs::read_to_string(file).ok()?;
        text.lines().find_map(|l| {
            let mut f = l.split(':');
            if f.next()? != name {
                return None;
            }
            let id = f.nth(1)?.parse().ok()?;
            let gid = f.next().and_then(|g| g.parse().ok()).unwrap_or(id);
            Some((id, gid))
        })
    };
    let not_found = |what: &str| io::Error::new(io::ErrorKind::NotFound, format!("unknown {what} {user}"));
    let (uid, primary_gid) = match u.parse::<u32>() {
        Ok(id) => (id, lookup_gid_for_uid(id).unwrap_or(id)),
        Err(_) => lookup("/etc/passwd", u).ok_or_else(|| not_found("user"))?,
    };
    let gid = match g {
        None => primary_gid,
        Some(g) => match g.parse::<u32>() {
            Ok(id) => id,
            Err(_) => lookup("/etc/group", g).map(|(id, _)| id).ok_or_else(|| not_found("group"))?,
        },
    };
    Ok((uid, gid))
}

fn lookup_gid_for_uid(uid: u32) -> Option<u32> {
    let text = fs::read_to_string("/etc/passwd").ok()?;
    text.lines().find_map(|l| {
        let f: Vec<&str> = l.split(':').collect();
        (f.get(2)?.parse::<u32>().ok()? == uid).then(|| f.get(3)?.parse().ok()).flatten()
    })
}

fn find_executable(p: &Prepared) -> Option<CString> {
    let name = p.argv[0].as_bytes();
    if name.contains(&b'/') {
        return Some(p.argv[0].clone());
    }
    p.path_dirs.iter().find_map(|dir| {
        let mut full = Vec::with_capacity(dir.len() + 1 + name.len());
        full.extend_from_slice(dir);
        full.push(b'/');
        full.extend_from_slice(name);
        let c = CString::new(full).ok()?;
        // SAFETY: valid NUL-terminated path.
        (unsafe { libc::access(c.as_ptr(), libc::X_OK) } == 0).then_some(c)
    })
}

#[cfg(test)]
mod tests {
    use kiri_proto::{Limits, Msg, Net, decode, encode_run, run_flags};

    /// A volume is filled from the image's directory as `apply_mount` reaches
    /// it: through `/proc/self/fd/N` of an O_PATH descriptor. The volume
    /// gets the directory's mode (not the link's 000) and its files.
    #[test]
    fn copy_tree_through_a_proc_fd_keeps_the_directory_mode() {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let base = std::env::temp_dir().join(format!("kiri-copy-tree-{}", std::process::id()));
        let (img, vol) = (base.join("img"), base.join("vol"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(img.join("sub")).unwrap();
        std::fs::create_dir_all(&vol).unwrap();
        std::fs::write(img.join("conf"), b"hello").unwrap();
        std::fs::set_permissions(img.join("conf"), std::fs::Permissions::from_mode(0o640)).unwrap();
        std::fs::set_permissions(img.join("sub"), std::fs::Permissions::from_mode(0o700)).unwrap();
        std::os::unix::fs::symlink("conf", img.join("link")).unwrap();
        std::fs::set_permissions(&img, std::fs::Permissions::from_mode(0o751)).unwrap();

        let fd = super::inroot::open_root(&img).unwrap();
        let through = std::path::PathBuf::from(format!("/proc/self/fd/{}", fd.as_raw_fd()));
        super::copy_tree(&through, &vol).unwrap();

        let mode = |p: &std::path::Path| std::fs::symlink_metadata(p).unwrap().mode() & 0o7777;
        assert_eq!(mode(&vol), 0o751, "the volume takes the image directory's mode");
        assert_eq!(std::fs::metadata(&vol).unwrap().uid(), std::fs::metadata(&img).unwrap().uid());
        assert_eq!(std::fs::read(vol.join("conf")).unwrap(), b"hello");
        assert_eq!(mode(&vol.join("conf")), 0o640);
        assert_eq!(mode(&vol.join("sub")), 0o700);
        assert_eq!(std::fs::read_link(vol.join("link")).unwrap(), std::path::Path::new("conf"));
        std::fs::remove_dir_all(&base).unwrap();
    }

    fn resolv(flags: u32, opts: &[&str], upstream: &[[u8; 4]]) -> String {
        let mut buf = Vec::new();
        let net = Net { bridge: "kiri1", ip: [10, 5, 0, 2], prefix: 24, gateway: [10, 5, 0, 1], extra_hosts: &[], dns_names: &[], opts };
        encode_run(&mut buf, "c1", &[], &["sh"], &[], "/", "", "c1", flags, net, &[], &Limits::default(), 0, &[] as &[&str]);
        let Ok(Msg::Run(spec)) = decode(&buf) else { panic!() };
        super::resolv_conf(&spec, upstream, (flags & run_flags::DNS_GATEWAY != 0).then_some(super::super::dns::EMBEDDED))
    }

    #[test]
    fn resolv_conf_follows_the_dns_settings() {
        let opts = ["nameserver=8.8.8.8", "search=corp.example", "search=.", "option=ndots:2", "option=timeout:1"];
        // A user network keeps the gateway's DNS, which forwards to 8.8.8.8.
        assert_eq!(resolv(run_flags::DNS_GATEWAY, &opts, &[[192, 168, 64, 1]]), "nameserver 127.0.0.11\nsearch corp.example\noptions ndots:2 timeout:1\n");
        // The default bridge and host networks get the servers themselves.
        assert_eq!(resolv(0, &opts, &[[192, 168, 64, 1]]), "nameserver 8.8.8.8\nsearch corp.example\noptions ndots:2 timeout:1\n");
        assert_eq!(resolv(0, &[], &[[192, 168, 64, 1]]), "nameserver 192.168.64.1\n");
    }

    #[test]
    fn a_resolver_at_one_of_the_hosts_own_addresses_goes_through_the_agent() {
        let mut buf = Vec::new();
        let net = Net { bridge: "kiri0", ip: [172, 17, 0, 2], prefix: 16, gateway: [172, 17, 0, 1], extra_hosts: &[], dns_names: &[], opts: &[] };
        encode_run(&mut buf, "c1", &[], &["sh"], &[], "/", "", "c1", 0, net, &[], &Limits::default(), 0, &[] as &[&str]);
        let Ok(Msg::Run(spec)) = decode(&buf) else { panic!() };
        // TEST-NET-3: no machine running the tests has it.
        assert!(!super::embedded_dns(&spec, &[[203, 0, 113, 53]], false), "a resolver elsewhere is asked directly");
        assert!(!super::embedded_dns(&spec, &[], false), "no resolver: the public fallback, not the agent");
    }

    #[test]
    fn in_the_vm_the_default_bridge_goes_through_the_agent() {
        let spec_with = |flags: u32, opts: &[&str], f: &dyn Fn(&kiri_proto::RunSpec<'_>)| {
            let mut buf = Vec::new();
            let net = Net { bridge: "kiri0", ip: [172, 17, 0, 2], prefix: 16, gateway: [172, 17, 0, 1], extra_hosts: &[], dns_names: &[], opts };
            encode_run(&mut buf, "c1", &[], &["sh"], &[], "/", "", "c1", flags, net, &[], &Limits::default(), 0, &[] as &[&str]);
            let Ok(Msg::Run(spec)) = decode(&buf) else { panic!() };
            f(&spec);
        };
        let gateway = [[192, 168, 64, 1]];
        spec_with(0, &[], &|s| assert!(super::embedded_dns(s, &gateway, true), "so a refused gateway can fall back to the host"));
        spec_with(0, &["nameserver=8.8.8.8"], &|s| assert!(!super::embedded_dns(s, &gateway, true), "--dns servers are asked directly"));
        spec_with(run_flags::NET_HOST, &[], &|s| assert!(!super::embedded_dns(s, &gateway, true), "host networking keeps the VM's resolv.conf"));
        // Named by the bridge's gateway, not 127.0.0.11: builders keep it
        // for their steps rather than swapping in 8.8.8.8.
        spec_with(0, &[], &|s| assert_eq!(super::embedded_address(s, true), [172, 17, 0, 1]));
        spec_with(0, &[], &|s| assert_eq!(super::resolv_conf(s, &gateway, Some(super::embedded_address(s, true))), "nameserver 172.17.0.1\n"));
        // A user network keeps Docker's 127.0.0.11, as does a Linux host.
        spec_with(run_flags::DNS_GATEWAY, &[], &|s| assert_eq!(super::embedded_address(s, true), super::super::dns::EMBEDDED));
        spec_with(0, &[], &|s| assert_eq!(super::embedded_address(s, false), super::super::dns::EMBEDDED));
    }

    #[test]
    fn macs_parse() {
        assert_eq!(super::net::parse_mac("02:42:ac:11:00:02"), Some([2, 0x42, 0xac, 0x11, 0, 2]));
        assert_eq!(super::net::mac_for([172, 17, 0, 2]), [2, 0x42, 0xac, 0x11, 0, 2]);
        assert_eq!(super::net::parse_mac("02:42:ac:11:00"), None);
        assert_eq!(super::net::parse_mac("02:42:ac:11:00:02:03"), None);
    }
}
